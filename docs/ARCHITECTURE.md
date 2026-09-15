# Model and Runtime Architecture

## Components

The runtime has two trained neural networks followed by a parametric body model:

1. **IMUCoCo** maps sparse, placement-aware IMU streams to 24 joint features.
2. **DTP Poser** maps those temporal features to pose, contact, and translation.
3. **SMPL/SMPL-H** converts rotations and translation into joints and a mesh.

## IMUCoCo

For each of 24 SMPL target joints, the loss map chooses one of the five available
sensors. The selected `[B,T,1,9]` stream enters a target-specific path:

```text
9D signal
 -> 9-to-128 linear projection and ReLU
 -> two 128-unit MFE LSTM layers
 -> placement modulation from the SCE
 -> three 128-unit JNM LSTM layers
 -> one 128D target-joint feature
```

The Sensor Coordinate Encoder normalizes the sensor coordinate relative to the
target joint, applies sine/cosine frequency encoding, adds a learned body-region
embedding, and uses an MLP to produce gamma/beta modulation for all three JNM
layers. The 24 paths run in parallel; an input does not traverse all 120 IMUCoCo
LSTM layers sequentially. Each path has five recurrent layers.

## DTP Poser

DTP receives `[B,T,24,128]`. A three-layer global LSTM creates shared context.
Three regional branches process upper body, lower body, and torso. Each regional
branch first estimates selected joint velocities and then global 6D rotations.
Separate two-layer recurrent heads estimate foot contact and root velocity.

The 6D rotations are converted to proper 3x3 matrices. SMPL inverse kinematics
produces local rotations, and forward kinematics produces joints. Translation
combines predicted root velocity with foot-contact-aware motion before floor
penetration correction.

## Online State

The network is causal and processes one frame at a time. MFE, JNM, pose,
contact, and velocity hidden states persist between calls. They are cleared
after recalibration or reconnection so stale motion history cannot contaminate
a new session.

## Tensor Contract

```text
synchronized frame                     [5,9]
online batch                            [1,1,5,9]
selected stream per target joint       [B,T,1,9]
MFE/JNM output per target joint        [B,T,1,128]
combined IMUCoCo output                [B,T,24,128]
DTP global and local rotations         [B,T,24,3,3]
DTP translation                        [T,3]
SMPL joints                            [T,24,3]
SMPL mesh                              [T,6890,3]
```

## Why These Models

IMUCoCo was chosen because sensor coordinates and its learned transfer-loss map
support sparse, flexible on-body placement. DTP was released with IMUCoCo for
human-pose estimation and supplies temporal regional pose, contact, and root
translation. LSTMs are appropriate for causal streaming because ambiguity in a
single IMU frame can be reduced using motion history without future frames.

SMPL provides a standardized 24-joint kinematic body. SMPL-H is used only for a
more complete visible mesh; the five-sensor network does not predict fingers.
