"""Host-side mission service for the g1_auto_navigation stack.

The System-2 loop runs HERE -- on the operator's machine, beside the MQTT
broker -- not in a browser tab and not on the robot. What that buys:

  * the model key stays in one process that is not a web page;
  * a mission outlives the tab that started it, and `./g1 mission` can drive
    the same loop from a terminal;
  * the Jetson gains nothing to schedule (it has been measured at load ~11 on
    8 cores with the navigation stack alone).

The robot is reached exactly the way the dashboard reaches it: CBOR over MQTT,
through the on-robot fleet agent, using the topics already in
`src/g1_fleet/config/topics.json`. Nothing here opens a DDS participant, and
nothing here can publish `/cmd_vel` or `/estop` -- see `wire.PUBLISHABLE`.
"""
