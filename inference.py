import os
import re
from argparse import ArgumentParser

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

try:
    from vllm import SamplingParams, LLM
    from qwen_vl_utils import process_vision_info
    VLLM_AVAILABLE = True
except ImportError:
    print("Unable to import VLLM")
    VLLM_AVAILABLE = False


def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', type=str,
                       default='Qwen/Qwen3-VL-2B-Instruct',
                       help='Checkpoint name or path')
    parser.add_argument('--cpu-only', action='store_true',
                       help='Run with CPU only')
    parser.add_argument('--flash-attn2', action='store_true',
                       help='Enable flash_attention_2')
    parser.add_argument('--backend', type=str, choices=['hf', 'vllm'],
                       default='vllm' if VLLM_AVAILABLE else 'hf',
                       help='Backend to use')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.70,
                       help='GPU memory utilization for vLLM')
    parser.add_argument('--tensor-parallel-size', type=int, default=None,
                       help='Tensor parallel size for vLLM')
    parser.add_argument('--max-model-len', type=int, default=20000,
                       help='Maximum model length for vLLM')
    return parser.parse_args()


def _load_model_processor(args):
    if args.backend == 'vllm':
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM not available. Install vllm and qwen-vl-utils.")
        
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        tensor_parallel_size = args.tensor_parallel_size
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
        
        model = LLM(
            model=args.checkpoint_path,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=False,
            tensor_parallel_size=tensor_parallel_size,
            seed=0
        )
        processor = AutoProcessor.from_pretrained(args.checkpoint_path)
        return model, processor, 'vllm'
    else:
        device_map = 'cpu' if args.cpu_only else 'auto'
        
        if args.flash_attn2:
            model = AutoModelForImageTextToText.from_pretrained(
                args.checkpoint_path,
                torch_dtype='auto',
                attn_implementation='flash_attention_2',
                device_map=device_map
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                args.checkpoint_path,
                device_map=device_map
            )
        
        processor = AutoProcessor.from_pretrained(args.checkpoint_path)
        return model, processor, 'hf'


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def _is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)


def _transform_messages(original_messages):
    transformed_messages = []
    for message in original_messages:
        new_content = []
        for item in message['content']:
            if 'image' in item:
                new_item = {'type': 'image', 'image': item['image']}
            elif 'text' in item:
                new_item = {'type': 'text', 'text': item['text']}
            elif 'video' in item:
                new_item = {'type': 'video', 'video': item['video']}
            else:
                continue
            new_content.append(new_item)
        new_message = {'role': message['role'], 'content': new_content}
        transformed_messages.append(new_message)
    return transformed_messages


def _prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def generate_response(model, processor, messages, backend):
    messages = _transform_messages(messages)
    
    if backend == 'vllm':
        inputs = _prepare_inputs_for_vllm(messages, processor)
        sampling_params = SamplingParams(max_tokens=1024, temperature=0.7)
        
        output = model.generate(inputs, sampling_params=sampling_params)
        response = ''
        for completion in output[0].outputs:
            response += completion.text
        
        return _remove_image_special(response)
    else:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"]

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return _remove_image_special(response)


def main():
    args = _get_args()
    print(f"Loading model from {args.checkpoint_path}...")
    model, processor, backend = _load_model_processor(args)
    print(f"Model loaded successfully! Backend: {backend.upper()}")
    print("=" * 60)
    print("Qwen3-VL REPL")
    print("Commands:")
    print("  /image <path>  - Add an image to the conversation")
    print("  /video <path>  - Add a video to the conversation")
    print("  /clear        - Clear conversation history")
    print("  /quit or /exit - Exit the REPL")
    print("=" * 60)
    print()
    
    messages = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.startswith('/quit') or user_input.startswith('/exit'):
                print("Goodbye!")
                break
            
            if user_input.startswith('/clear'):
                messages = []
                print("Conversation history cleared.")
                continue
            
            if user_input.startswith('/image '):
                image_path = user_input[7:].strip()
                if os.path.exists(image_path):
                    messages.append({
                        'role': 'user',
                        'content': [{'type': 'image', 'image': os.path.abspath(image_path)}]
                    })
                    print(f"Image added: {image_path}")
                else:
                    print(f"Error: File not found: {image_path}")
                continue
            
            if user_input.startswith('/video '):
                video_path = user_input[7:].strip()
                if os.path.exists(video_path):
                    messages.append({
                        'role': 'user',
                        'content': [{'type': 'video', 'video': os.path.abspath(video_path)}]
                    })
                    print(f"Video added: {video_path}")
                else:
                    print(f"Error: File not found: {video_path}")
                continue
            
            # Add text message
            content = [{'type': 'text', 'text': user_input}]
            messages.append({'role': 'user', 'content': content})
            
            print("Assistant: ", end='', flush=True)
            response = generate_response(model, processor, messages, backend)
            print(response)
            print()
            
            # Add assistant response to history
            messages.append({
                'role': 'assistant',
                'content': [{'type': 'text', 'text': response}]
            })
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Use /quit to exit.")
        except EOFError:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()