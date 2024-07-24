# SPoC Decoding
This is the dareplane version of the bollinger control. It takes a LSL stream with a single channel, calculates the bands, broadcasts the band values to LSL and sends out control requests.
## General remarks
There is still a open point for consideration:
    - [ ] How will the control request (e.g. use stimulation, shut down stim) be sent to the i/o module for stimulation. It could either be sent to the socket of the i/o module, or just send the control variable
    as another LSL stream. This would then needs to be parsed by the i/o module...

