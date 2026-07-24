"""Assemble the LEAP-hand lift scene.

The menagerie LEAP hand is a fixed-base 16-DoF hand. We bolt it onto a 4-DoF
actuated wrist (x / y / z slides + a yaw hinge) so it can actually approach and
lift something, and drop a cube on a table underneath it.

Everything is built with MjSpec rather than a hand-written XML so the hand model
stays a pristine include from mujoco_menagerie.
"""

from pathlib import Path

import mujoco
import numpy as np

MENAGERIE = Path("/home/leo/gello_software/third_party/mujoco_menagerie")
LEAP_XML = MENAGERIE / "leap_hand" / "right_hand.xml"

# Wrist joint limits (metres, radians). The hand hangs palm-down from the wrist.
WRIST_RANGES = {
    "wrist_x": (-0.18, 0.18),
    "wrist_y": (-0.18, 0.18),
    "wrist_z": (-0.09, 0.12),
    "wrist_yaw": (-1.6, 1.6),
}
WRIST_HOME = np.array([0.0, 0.0, 0.0, 0.0])

CUBE_HALF = 0.028  # 5.6 cm cube -- a comfortable LEAP-sized object
TABLE_HEIGHT = 0.0
WRIST_MOUNT_Z = 0.28

# The stock palm carries a 180-deg flip about x (the menagerie scene shows the
# hand palm-up). Undoing it with the same quat leaves the palm facing straight
# down: fingers reach out along +x and curl *downwards*, which is the classic
# top-down power grasp. The offset re-centres the closing volume under the wrist.
HAND_ATTACH_QUAT = (0.0, 1.0, 0.0, 0.0)
HAND_ATTACH_POS = (0.027, 0.025, 0.0)
# Where the fingers and thumb actually meet, relative to the wrist origin.
GRASP_OFFSET_Z = 0.178
# Fingertips hang this far below the wrist origin when the hand is fully open.
FINGERTIP_DROP = 0.122

N_HAND_JOINTS = 16
N_WRIST_JOINTS = 4
N_ACT = N_HAND_JOINTS + N_WRIST_JOINTS


def build_spec() -> mujoco.MjSpec:
    """Return the full scene spec: table + cube + wrist-mounted LEAP hand."""
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 100.0
    spec.option.timestep = 0.002

    _add_visuals(spec)
    _add_table(spec)
    _add_cube(spec)
    _add_hand(spec)

    return spec


def _add_visuals(spec: mujoco.MjSpec) -> None:
    spec.stat.center = [0.0, 0.0, 0.15]
    spec.stat.extent = 0.6
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    spec.visual.global_.azimuth = 140
    spec.visual.global_.elevation = -25

    spec.add_texture(
        name="skybox",
        type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
        rgb1=[0.3, 0.5, 0.7],
        rgb2=[0.0, 0.0, 0.0],
        width=512,
        height=3072,
    )
    spec.add_texture(
        name="grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        mark=mujoco.mjtMark.mjMARK_EDGE,
        rgb1=[0.22, 0.25, 0.30],
        rgb2=[0.16, 0.19, 0.24],
        markrgb=[0.75, 0.75, 0.75],
        width=300,
        height=300,
    )
    mat = spec.add_material(name="grid", texrepeat=[4, 4], texuniform=True, reflectance=0.15)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"

    spec.add_material(name="cube_mat", rgba=[0.85, 0.35, 0.25, 1.0], specular=0.3, shininess=0.4)

    spec.worldbody.add_light(
        pos=[0, 0, 1.2], dir=[0, 0, -1], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    )
    # A fixed camera that frames the workspace, for rollout videos.
    spec.worldbody.add_camera(
        name="track",
        pos=[0.45, -0.45, 0.42],
        xyaxes=[0.7, 0.7, 0.0, -0.28, 0.28, 0.92],
    )


def _add_table(spec: mujoco.MjSpec) -> None:
    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0.6, 0.6, 0.05],
        pos=[0, 0, TABLE_HEIGHT],
        material="grid",
        condim=3,
        friction=[1.0, 0.02, 0.001],
    )


def _add_cube(spec: mujoco.MjSpec) -> None:
    cube = spec.worldbody.add_body(name="cube", pos=[0.0, 0.0, CUBE_HALF])
    cube.add_freejoint(name="cube_free")
    cube.add_geom(
        name="cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[CUBE_HALF] * 3,
        material="cube_mat",
        mass=0.09,
        condim=4,
        friction=[1.2, 0.02, 0.001],
        priority=1,
    )
    cube.add_site(name="cube_center", size=[0.004], rgba=[0, 0, 0, 0])


def _add_hand(spec: mujoco.MjSpec) -> None:
    """Attach the menagerie hand under a 4-DoF actuated wrist."""
    hand = mujoco.MjSpec.from_file(str(LEAP_XML))

    # The stock model has no free joint, so the palm attaches rigidly to our wrist.
    wrist = spec.worldbody.add_body(name="wrist", pos=[0.0, 0.0, WRIST_MOUNT_Z])

    axes = {"wrist_x": [1, 0, 0], "wrist_y": [0, 1, 0], "wrist_z": [0, 0, 1]}
    for name, axis in axes.items():
        wrist.add_joint(
            name=name,
            type=mujoco.mjtJoint.mjJNT_SLIDE,
            axis=axis,
            range=WRIST_RANGES[name],
            damping=12.0,
            armature=0.02,
        )
    wrist.add_joint(
        name="wrist_yaw",
        type=mujoco.mjtJoint.mjJNT_HINGE,
        axis=[0, 0, 1],
        range=WRIST_RANGES["wrist_yaw"],
        damping=0.6,
        armature=0.01,
    )
    # Cosmetic only (no contacts). Sits directly on the back of the palm, which
    # the attach frame places 10 cm below the wrist origin.
    wrist.add_geom(
        name="wrist_mount",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.042, 0.048, 0.011],
        pos=[HAND_ATTACH_POS[0], HAND_ATTACH_POS[1], -0.087],
        rgba=[0.25, 0.27, 0.32, 1.0],
        contype=0,
        conaffinity=0,
        mass=0.2,
    )
    wrist.add_site(name="wrist_center", size=[0.004], rgba=[0, 0, 0, 0])

    frame = wrist.add_frame(pos=list(HAND_ATTACH_POS), quat=list(HAND_ATTACH_QUAT))
    spec.attach(hand, prefix="", frame=frame)

    # Wrist actuators: position servos matching the hand's control style.
    # wrist_z is stiff on purpose. At kp=1200 the ~10 N of hand weight produced
    # 8 mm of steady-state droop, so the hand never actually reached its commanded
    # height -- which silently broke any controller with a height tolerance
    # tighter than the sag.
    gains = {"wrist_x": 2500.0, "wrist_y": 2500.0, "wrist_z": 8000.0, "wrist_yaw": 60.0}
    for name in ("wrist_x", "wrist_y", "wrist_z", "wrist_yaw"):
        spec.add_actuator(
            name=f"{name}_act",
            target=name,
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gainprm=[gains[name]] + [0.0] * 9,
            biasprm=[0.0, -gains[name], -gains[name] * 0.06] + [0.0] * 7,
            biastype=mujoco.mjtBias.mjBIAS_AFFINE,
            ctrlrange=WRIST_RANGES[name],
            ctrllimited=mujoco.mjtLimited.mjLIMITED_TRUE,
        )


def compile_model() -> mujoco.MjModel:
    return build_spec().compile()


if __name__ == "__main__":
    m = compile_model()
    print(f"nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody}")
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    print("actuators:", names)
    jnames = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    print("joints:", jnames)
