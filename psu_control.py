import pyvisa, time, csv

# --- modulation (slow) — this is what the lock-in extracts ---
F_MOD  = 0.1          # Hz
DUTY   = 0.50          # fraction of each period with power on
CYCLES = 20

# --- carrier (fast) — what the heater actually runs on ---
F_LINE = 400.0         # Hz
V_OP   = 75
I_LIM  = 20.0          # A — set above expected draw, below anything alarming

PERIOD = 1.0 / F_MOD
ON     = PERIOD * DUTY
OFF    = PERIOD - ON
print(f"period {PERIOD:.1f} s -> {ON:.1f} s on, {OFF:.1f} s off")
print(f"total run {CYCLES * PERIOD / 60:.1f} min")

rm  = pyvisa.ResourceManager()
psu = rm.open_resource('USB0::0x0A69::0x0883::96160900000067::INSTR')
print(psu.query('*IDN?'))
psu.write('*CLS')                  # drop any stale errors left in the queue

# --- configure: runs once, before the loop ---
psu.write(f'FREQ {F_LINE}')
psu.write(f'CURR:LIM {I_LIM}')
psu.write('OUTP:COUP AC')
psu.write('VOLT:AC 0')             # start at zero so enabling is uneventful
err = psu.query('SYST:ERR?').strip()
if not err.startswith('0,'):       # instrument returns e.g. 0,"No error"
    psu.close()
    raise RuntimeError(f'PSU rejected configuration: {err}')
psu.write('OUTP ON')

start = time.monotonic()

with open('excitation_log.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t_s', 'cycle', 'state', 'current_A'])
    try:
        for i in range(CYCLES):
            for state, volts, dur in (('on', V_OP, ON), ('off', 0.0, OFF)):
                psu.write(f'VOLT:AC {volts}')
                target = time.monotonic() + dur
                while time.monotonic() < target:
                    w.writerow([round(time.monotonic() - start, 3), i, state,
                                psu.query('MEAS:CURR:AC?').strip()])
                    f.flush()
                    time.sleep(1.0)
    finally:
        psu.write('VOLT:AC 0')
        psu.write('OUTP OFF')
        psu.close()
        print("output off")