#!/usr/bin/python3
"""Inert Splash v5 peer. Does not import any model, tokenizer, or GPU library."""
import os
import struct
import sys
import time

mode = os.path.basename(os.path.dirname(sys.argv[2]))
header = struct.Struct("<4sHHHHQI")


def send(kind, payload):
    frame = header.pack(b"SPLH", 5, 24, kind, 0, len(payload), 0) + payload
    # Deliberately fragmented writes exercise the stream reader.
    for offset in range(0, len(frame), 3):
        sys.stdout.buffer.write(frame[offset:offset + 3])
        sys.stdout.buffer.flush()


if mode == "startup-timeout":
    time.sleep(30)
    sys.exit(0)
if mode == "truncated":
    sys.stdout.buffer.write(b"SPLH")
    sys.stdout.buffer.flush()
    sys.exit(0)
send(0x100, struct.pack("<QIIQ", 1, 1, 4096, 15))
while True:
    raw = sys.stdin.buffer.read(24)
    if not raw:
        break
    magic, version, size, kind, flags, count, reserved = header.unpack(raw)
    assert (magic, version, size, flags, reserved) == (b"SPLH", 5, 24, 0, 0)
    payload = sys.stdin.buffer.read(count)
    if kind == 2:
        break
    assert kind == 1
    fields = struct.unpack_from("<QBBBQQIIIffIQB", payload)
    request_id, priority, cohort, constraint, absolute, remaining, limit, prompts, images, temperature, top_p, top_k, seed, progress = fields
    assert priority == 1 and images == 0 and constraint == 0
    assert len(payload) == 60 + 4 * prompts
    assert struct.unpack_from("<II", payload, 60) == (7, 8)
    if mode == "wait":
        # Await Cancel rather than producing a token.
        continue
    if mode == "error":
        code, message = b"fixture_error", b"CPU peer rejected request"
        send(0x105, struct.pack("<BBQII", 1, 0, request_id, len(code), len(message)) + code + message)
        continue
    if mode == "bad-id":
        request_id += 1
    send(0x101, struct.pack("<QBiII", request_id, 0, 0, 0, 4096))
    send(0x102, struct.pack("<QIII", request_id, 0, 1, 65))
    send(0x102, struct.pack("<QIII", request_id, 1, 1, 66))
    send(0x104, struct.pack("<QBIIQQQ", request_id, 0, prompts, 2, 1, 1, 2))
