import pyaudio

p = pyaudio.PyAudio()

print("=== Dispositivos de ENTRADA (inputs) ===")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    max_in = int(info.get("maxInputChannels", 0) or 0)
    if max_in > 0:
        name = info.get("name", "")
        rate = info.get("defaultSampleRate", "")
        print(f"[{i:02d}] in={max_in}  rate={rate}  name={name}")

try:
    default_in = p.get_default_input_device_info()
    print("\nDefault input:")
    print(f" -> index={default_in['index']} name={default_in['name']} rate={default_in.get('defaultSampleRate')}")
except Exception as e:
    print("\nNo default input:", e)

p.terminate()
