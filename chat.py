from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from curtsies.fmtfuncs import *
import argparse
import os
import queue
import sounddevice as sd
import vosk
import sys
import json

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model_gpt = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")

q = queue.Queue()

def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(bytes(indata))

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-f', '--filename', type=str, metavar='FILENAME',
    help='audio file to store recording to')
parser.add_argument(
    '-m', '--model', type=str, metavar='MODEL_PATH',
    help='Path to the model')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-r', '--samplerate', type=int, help='sampling rate')
args = parser.parse_args(remaining)

try:
    if args.model is None:
        args.model = "model"
    if not os.path.exists(args.model):
        print ("Please download a model for your language from https://alphacephei.com/vosk/models")
        print ("and unpack as 'model' in the current folder.")
        parser.exit(0)
    if args.samplerate is None:
        device_info = sd.query_devices(args.device, 'input')
        # soundfile expects an int, sounddevice provides a float:
        args.samplerate = int(device_info['default_samplerate'])

    model = vosk.Model(args.model)

    if args.filename:
        dump_fn = open(args.filename, "wb")
    else:
        dump_fn = None

    with sd.RawInputStream(samplerate=args.samplerate, blocksize = 8000, device=args.device, dtype='int16',
                            channels=1, callback=callback):
            print('#' * 80)
            print('Press Ctrl+C to stop the recording')
            print('#' * 80)

            rec = vosk.KaldiRecognizer(model, args.samplerate)
            counter = 0
            while True:
                data = q.get()
                if rec.AcceptWaveform(data):
                    output = json.loads(rec.Result())
                   
                    print(green(f">> User:  {output['text']} "))
                    # encode the new user input, add the eos_token and return a tensor in Pytorch
                    new_user_input_ids = tokenizer.encode(output['text'] + tokenizer.eos_token, return_tensors='pt')
                    if counter == 0:
                        bot_input_ids = torch.cat([new_user_input_ids], dim=-1)
                    else:
                    # append the new user input tokens to the chat history
                        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

                    # generated a response while limiting the total chat history to 1000 tokens, 
                    chat_history_ids = model_gpt.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

                    # pretty print last ouput tokens from bot
                    print(red("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))))
                if dump_fn is not None:
                    dump_fn.write(data)

    
except KeyboardInterrupt:
    print('\nDone')
    parser.exit(0)
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))