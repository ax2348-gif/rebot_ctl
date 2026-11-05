"""Command-line tools, installed as console scripts with the package.

With the venv active these are on PATH:

    rebot_setup   check the arm's one-time setup (motor CAN watchdog, feedback, joint limits);
                  ``--store-watchdog`` writes the watchdog to motor flash
    rebot_zero    set the current pose as every joint's zero
    rebot_vis     all joints slack (MIT, zero gains) and the live pose in MeshCat
    rebot_jog     jog the end effector from the keyboard (IK; defaults to the safe config)

Each takes an optional hardware config file (default: ``hardware_yaml`` in
config/rebotarm.yaml) and ``--help``.
"""
