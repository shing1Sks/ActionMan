"""ActionMan: decision encoder + masked-diffusion argument filler."""
import os
import time


def beat():
    """Touch $HEARTBEAT (if set) so the box watchdog knows we're alive."""
    p = os.environ.get("HEARTBEAT")
    if p:
        with open(p, "w") as f:
            f.write(str(time.time()))
