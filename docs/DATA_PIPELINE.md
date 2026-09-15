# Data Pipeline

## Input

MQTT batches contain sequence, timestamp, acceleration XYZ, gyroscope XYZ, and
quaternion WXYZ. Direct serial input expects comma-separated GlobalPose packets:

```text
node_id,timestamp_ms,sequence,ax,ay,az,qw,qx,qy,qz,gx,gy,gz
```

## Processing Order

```text
MQTT JSON parsing or serial framing
 -> field and finite-value validation
 -> quaternion normalization and hemisphere continuity
 -> five-node grouping and host timestamps (MQTT)
 -> fixed mounting and model-axis transforms
 -> T-pose orientation reference
 -> gravity measurement
 -> acceleration bias/noise measurement
 -> gyro-based stationary detection
 -> stationary-only adaptive bias
 -> acceleration EMA and adaptive deadband
 -> per-sensor buffering
 -> quaternion SLERP and linear vector interpolation
 -> 6D orientation conversion
 -> [1,1,5,9] PyTorch tensor
 -> IMUCoCo and DTP inference
```

The model frame is right-handed: +X subject-left, +Y up, and +Z forward. The
ankle hardware X channels are converted into the pretrained model signal basis.

For rotation matrix `R`, model orientation channels are:

```text
[R00,R10,R20,R01,R11,R21]
```

The remaining three channels are calibrated global linear acceleration in
metres per second squared. No acceleration double integration is used to create
joint positions.

## Calibration and Filtering

Defaults are two seconds of T-pose reference, two seconds of gravity capture,
and five seconds of residual bias/noise capture. The calibration gyro limit is
3 deg/s. Live stationary detection remains stricter at 1 deg/s for 0.35 seconds.

Acceleration uses EMA alpha 0.25 and per-axis deadband:

```text
max(0.05 m/s^2, 3 * measured_noise_standard_deviation)
```

Any startup sensor loss invalidates calibration. Recalibration also clears
recurrent neural-network state.
