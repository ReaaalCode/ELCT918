# ELCT918 Verilog Refresher Assignment

Files:
- `design.sv`: original `mystery` module followed by a readable behavioral reconstruction.
- `testbench.sv`: directed experiments, a 10,000-cycle randomized differential test, and password demonstration.
- `report.md`: derivation, state/output equations, observations, and expected simulator transcript.

## Key

The plaintext nibble sequence is `6, 3, B, 9` (hex `63B9`). It produces ciphertext nibbles `C, A, F, E` (hex `CAFE`) and asserts `flag` on the fourth nibble.

## Run Results

Expected final result: `PASS: 10012 cycles compared, zero mismatches`.

Run it on:
- EDA Playground: `[https://edaplayground.com/x/Cuch]`
