# -*- coding: utf-8 -*-
"""
Abaqus 2024 一键搭建第一级定轴齿轮副显式动力学仿真工程
自动构建并校验两套完整模型与作业：
  1. Model-Healthy / Job-Healthy : SS1.5-29 <-> SS1.5-95 (健康基准)
  2. Model-Missing / Job-Missing : SS1.5-29 <-> SS1.5-95 (第 15 号齿单齿全缺)

核心参数（统一 mm - tonne - s - N - MPa 单位制）：
  - 齿轮：m = 1.5 mm, alpha = 20 deg, 齿宽 = 15 mm, 中心距 = 93.0 mm
  - 材料：S45C (E = 206000 MPa, nu = 0.3, rho = 7.85e-9 tonne/mm^3)
  - 支承：ER-16K 弹性轴承接地 Cartesian 连接器 (CONN2D2)
          RP-1: kx=ky=1e5 N/mm, cx=cy=0.7 N*s/mm
          RP-2: kx=ky=1e5 N/mm, cx=cy=1.5 N*s/mm
  - 工况：30 Hz (1800 rpm, VR3 = 188.5 rad/s, 前 0.03 s Smooth Step 升速)
          从动轮等效阻力矩 CM3 = +10000 N*mm (10 N*m)
  - 采样：RP-1 / RP-2 历程输出 12800 Hz (dt = 7.8125e-5 s), A1/A2/V1/V2/U1/U2/RF1/RF2
          轴承连接器支反力 CTF1/CTF2 @ 12800 Hz
  - 加速：局部微量半自动质量缩放 (BELOW_MIN, target dt = 8.0e-8 s, 仅作用于极少数齿根小单元)
          外齿廓定向接触配对 (排除内孔与自接触)
"""
import math
import os
import mesh
import connectorBehavior
from part import EdgeArray
from abaqus import mdb
from abaqusConstants import *


# ======================== 全局仿真配置 ========================
WORKDIR = r'D:\Dev\abaSimu\abaSimulation'
CAE_FILENAME = 'stage1_gearbox.cae'

MODULE_M = 1.5
ALPHA_DEG = 20.0
GEAR_WIDTH = 15.0
BORE_RADIUS = 12.7
CENTER_DIST = 93.0

STEP_TIME = 0.5          # 正式仿真时长 0.5 s（含 4~5 次缺齿冲击）
RAMP_TIME = 0.03         # Smooth Step 升速时间
INPUT_OMEGA = 188.5      # 30 Hz = 1800 rpm = 188.5 rad/s
LOAD_TORQUE = 10000.0    # +10 N*m = +10000 N*mm（逆时针阻碍顺时针从动轮）
HIST_DT = 7.8125e-5      # 12800 Hz 采样间隔
FIELD_INTERVALS = 100    # 100 帧场输出（每帧 0.005 s）
TARGET_DT = 8.0e-8       # 局部微量质量缩放目标步长（仅缩放低于此值的小单元）
NUM_CPUS = 8             # 1.5 万单元最优并行核数


def involute_gear_points(Z, m=MODULE_M, alpha_deg=ALPHA_DEG,
                         angle_offset=0.0, missing_tooth_idx=-1, n_flank=4):
    """生成闭环渐开线直齿轮外轮廓点列 (x, y)，单位 mm。"""
    alpha = math.radians(alpha_deg)
    r = 0.5 * m * Z
    rb = r * math.cos(alpha)
    ra = r + m
    rf = r - 1.25 * m
    step = 2.0 * math.pi / Z

    inv_a = math.tan(alpha) - alpha
    psi_b = math.pi / (2.0 * Z) + inv_a

    def inv_theta(rk):
        ak = math.acos(min(1.0, rb / rk))
        return psi_b - (math.tan(ak) - ak)

    r_start = max(rb, rf)
    th_start = inv_theta(r_start)
    th_root = min(th_start + 0.08 * step, 0.42 * step)

    pts = []
    for i in range(Z):
        a0 = i * step + angle_offset

        if i == missing_tooth_idx:
            for frac in (-0.42, -0.21, 0.0, 0.21, 0.42):
                ang = a0 + frac * step
                pts.append((rf * math.cos(ang), rf * math.sin(ang)))
            continue

        a_lr = a0 - th_root
        pts.append((rf * math.cos(a_lr), rf * math.sin(a_lr)))

        for k in range(n_flank + 1):
            rk = r_start + (ra - r_start) * k / float(n_flank)
            ang = a0 - inv_theta(rk)
            pts.append((rk * math.cos(ang), rk * math.sin(ang)))

        pts.append((ra * math.cos(a0), ra * math.sin(a0)))

        for k in range(n_flank, -1, -1):
            rk = r_start + (ra - r_start) * k / float(n_flank)
            ang = a0 + inv_theta(rk)
            pts.append((rk * math.cos(ang), rk * math.sin(ang)))

        a_rr = a0 + th_root
        pts.append((rf * math.cos(a_rr), rf * math.sin(a_rr)))

    return pts, rf


def create_and_mesh_gear(model, part_name, Z, angle_offset=0.0, missing_tooth_idx=-1):
    """创建 2D 渐开线齿轮部件、指派 S45C 15mm 截面并划分优化网格。"""
    pts, rf = involute_gear_points(
        Z=Z, angle_offset=angle_offset, missing_tooth_idx=missing_tooth_idx
    )
    sheet_size = max(200.0, MODULE_M * Z * 3.0)
    s = model.ConstrainedSketch(name='__profile_' + part_name, sheetSize=sheet_size)

    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) > 1e-6:
            s.Line(point1=p1, point2=p2)

    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(BORE_RADIUS, 0.0))

    p = model.Part(
        name=part_name,
        dimensionality=TWO_D_PLANAR,
        type=DEFORMABLE_BODY
    )
    p.BaseShell(sketch=s)
    del model.sketches['__profile_' + part_name]

    # 齿圈分区圆
    r_part = rf - (4.0 if Z < 50 else 8.5)
    if r_part > BORE_RADIUS + 2.0:
        sp = model.ConstrainedSketch(name='__part_' + part_name, sheetSize=sheet_size)
        sp.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r_part, 0.0))
        p.PartitionFaceBySketch(faces=p.faces[:], sketch=sp)
        del model.sketches['__part_' + part_name]

    # 指派截面
    fset = p.Set(faces=p.faces[:], name='Set_AllFaces')
    p.SectionAssignment(region=fset, sectionName='Sec_Gear_15mm')

    # 按半径分类边：内孔 / 分区圆 / 外齿廓
    r_part_max = 17.5 if Z < 50 else 63.0
    bore_edges = []
    part_edges = []
    tooth_edges = []
    for e in p.edges:
        pt = e.pointOn[0]
        r = math.hypot(pt[0], pt[1])
        if r < 14.0:
            bore_edges.append(e)
        elif r < r_part_max:
            part_edges.append(e)
        else:
            tooth_edges.append(e)

    # 在 Part 级别定义命名表面，装配后直接复用，避免坐标脆弱选取
    p.Surface(side1Edges=EdgeArray(bore_edges), name='Surf_Bore')
    p.Surface(side1Edges=EdgeArray(tooth_edges), name='Surf_Teeth')

    # 优化网格划分：外齿廓 0.45 mm、分区圆 1.5 mm、轮辐 3.0 mm
    p.seedPart(size=3.0, deviationFactor=0.1, minSizeFactor=0.5)
    if part_edges:
        p.seedEdgeBySize(edges=part_edges, size=1.5, deviationFactor=0.1,
                         minSizeFactor=0.5, constraint=FINER)
    if tooth_edges:
        p.seedEdgeBySize(edges=tooth_edges, size=0.45, deviationFactor=0.1,
                         minSizeFactor=0.5, constraint=FINER)

    p.setMeshControls(
        regions=p.faces[:],
        elemShape=QUAD_DOMINATED,
        technique=FREE,
        algorithm=ADVANCING_FRONT
    )
    elem_quad = mesh.ElemType(elemCode=CPS4R, elemLibrary=EXPLICIT)
    elem_tri = mesh.ElemType(elemCode=CPS3, elemLibrary=EXPLICIT)
    p.setElementType(regions=(p.faces[:],), elemTypes=(elem_quad, elem_tri))
    p.generateMesh()

    return p


def build_single_model(model_name, missing_tooth_idx=-1):
    """构建一套完整的显式动力学模型（健康或单齿缺失）。"""
    if model_name in mdb.models:
        del mdb.models[model_name]
    m = mdb.Model(name=model_name)

    # 1. 材料与截面
    mat = m.Material(name='S45C')
    mat.Density(table=((7.85e-9, ), ))
    mat.Elastic(table=((206000.0, 0.3), ))
    m.HomogeneousSolidSection(
        name='Sec_Gear_15mm',
        material='S45C',
        thickness=GEAR_WIDTH
    )

    # 2. 创建并划分两个齿轮部件
    p29 = create_and_mesh_gear(m, 'Gear_29', Z=29, angle_offset=0.0, missing_tooth_idx=-1)
    offset_95 = math.pi + math.pi / 95.0
    p95_name = 'Gear_95_Missing' if missing_tooth_idx >= 0 else 'Gear_95_Healthy'
    p95 = create_and_mesh_gear(
        m, p95_name, Z=95, angle_offset=offset_95, missing_tooth_idx=missing_tooth_idx
    )

    # 3. 装配定位与参考点
    a = m.rootAssembly
    a.DatumCsysByDefault(CARTESIAN)
    inst29 = a.Instance(name='Gear_29-1', part=p29, dependent=ON)
    inst95 = a.Instance(name=p95_name + '-1', part=p95, dependent=ON)
    a.translate(instanceList=(p95_name + '-1', ), vector=(CENTER_DIST, 0.0, 0.0))

    rp1_feat = a.ReferencePoint(point=(0.0, 0.0, 0.0))
    rp2_feat = a.ReferencePoint(point=(CENTER_DIST, 0.0, 0.0))
    rp1 = a.referencePoints[rp1_feat.id]
    rp2 = a.referencePoints[rp2_feat.id]

    set_rp1 = a.Set(referencePoints=(rp1, ), name='Set_RP1')
    set_rp2 = a.Set(referencePoints=(rp2, ), name='Set_RP2')

    # 4. 显式动力学分析步（含局部微量半自动质量缩放）
    m.ExplicitDynamicsStep(
        name='Step-1',
        previous='Initial',
        timePeriod=STEP_TIME,
        improvedDtMethod=ON,
        massScaling=((
            SEMI_AUTOMATIC, MODEL, AT_BEGINNING,
            0.0, TARGET_DT, BELOW_MIN,
            0, 0, 0.0, 0.0, 0, None
        ), )
    )

    # 场输出（精简变量防止 ODB 过大）
    m.fieldOutputRequests['F-Output-1'].setValues(
        variables=('S', 'U'),
        numIntervals=FIELD_INTERVALS
    )

    # 历程输出：12800 Hz 虚拟加速度计（RP2 缺齿轴 + RP1 输入轴）
    if 'H-Output-1' in m.historyOutputRequests:
        m.historyOutputRequests['H-Output-1'].setValues(
            variables=('ALLIE', 'ALLKE', 'ALLWK', 'ETOTAL'),
            numIntervals=200
        )
    m.HistoryOutputRequest(
        name='H-Output-RP2',
        createStepName='Step-1',
        variables=('A1', 'A2', 'V1', 'V2', 'U1', 'U2', 'RF1', 'RF2'),
        timeInterval=HIST_DT,
        region=set_rp2
    )
    m.HistoryOutputRequest(
        name='H-Output-RP1',
        createStepName='Step-1',
        variables=('A1', 'A2', 'V1', 'V2', 'U1', 'U2', 'RF1', 'RF2'),
        timeInterval=HIST_DT,
        region=set_rp1
    )

    # 5. 相互作用：接触属性 + 外齿廓定向通用接触
    ip = m.ContactProperty('IntProp-Gear')
    ip.TangentialBehavior(
        formulation=PENALTY,
        directionality=ISOTROPIC,
        slipRateDependency=OFF,
        pressureDependency=OFF,
        temperatureDependency=OFF,
        dependencies=0,
        table=((0.08, ), ),
        shearStressLimit=None,
        maximumElasticSlip=FRACTION,
        fraction=0.005,
        elasticSlipStiffness=None
    )
    ip.NormalBehavior(pressureOverclosure=HARD, allowSeparation=ON)
    ip.GeometricProperties(contactArea=GEAR_WIDTH, padThickness=None)

    gc = m.ContactExp(name='Int-Contact', createStepName='Initial')
    gc.includedPairs.setValuesInStep(
        stepName='Initial',
        useAllstar=OFF,
        addPairs=((inst29.surfaces['Surf_Teeth'], inst95.surfaces['Surf_Teeth']), )
    )
    gc.contactPropertyAssignments.appendInStep(
        stepName='Initial',
        assignments=((GLOBAL, SELF, 'IntProp-Gear'), )
    )

    # 6. 内孔运动学耦合（RP1 -> Gear_29 内孔, RP2 -> Gear_95 内孔）
    m.Coupling(
        name='Coup_29',
        controlPoint=set_rp1,
        surface=inst29.surfaces['Surf_Bore'],
        influenceRadius=WHOLE_SURFACE,
        couplingType=KINEMATIC,
        u1=ON, u2=ON, u3=OFF, ur1=OFF, ur2=OFF, ur3=ON
    )
    m.Coupling(
        name='Coup_95',
        controlPoint=set_rp2,
        surface=inst95.surfaces['Surf_Bore'],
        influenceRadius=WHOLE_SURFACE,
        couplingType=KINEMATIC,
        u1=ON, u2=ON, u3=OFF, ur1=OFF, ur2=OFF, ur3=ON
    )

    # 7. ER-16K 弹性轴承接地 Cartesian 连接器 (CONN2D2)
    c1_el = connectorBehavior.ConnectorElasticity(
        behavior=LINEAR, components=(1, 2), table=((100000.0, 100000.0), )
    )
    c1_da = connectorBehavior.ConnectorDamping(
        behavior=LINEAR, components=(1, 2), table=((0.7, 0.7), )
    )
    m.ConnectorSection(
        name='ConnSec_Bearing_RP1',
        translationalType=CARTESIAN,
        behaviorOptions=(c1_el, c1_da)
    )

    c2_el = connectorBehavior.ConnectorElasticity(
        behavior=LINEAR, components=(1, 2), table=((100000.0, 100000.0), )
    )
    c2_da = connectorBehavior.ConnectorDamping(
        behavior=LINEAR, components=(1, 2), table=((1.5, 1.5), )
    )
    m.ConnectorSection(
        name='ConnSec_Bearing_RP2',
        translationalType=CARTESIAN,
        behaviorOptions=(c2_el, c2_da)
    )

    w1 = a.WirePolyLine(points=((None, rp1), ), mergeType=IMPRINT, meshable=False)
    set_conn1 = a.Set(edges=a.getFeatureEdges(w1.name), name='Set_Conn_RP1')
    a.SectionAssignment(region=set_conn1, sectionName='ConnSec_Bearing_RP1')

    w2 = a.WirePolyLine(points=((None, rp2), ), mergeType=IMPRINT, meshable=False)
    set_conn2 = a.Set(edges=a.getFeatureEdges(w2.name), name='Set_Conn_RP2')
    a.SectionAssignment(region=set_conn2, sectionName='ConnSec_Bearing_RP2')

    # 输出轴承连接器支反力 CTF1, CTF2 @ 12800 Hz
    m.HistoryOutputRequest(
        name='H-Output-Bearing-RP2',
        createStepName='Step-1',
        variables=('CTF1', 'CTF2'),
        timeInterval=HIST_DT,
        region=set_conn2
    )
    m.HistoryOutputRequest(
        name='H-Output-Bearing-RP1',
        createStepName='Step-1',
        variables=('CTF1', 'CTF2'),
        timeInterval=HIST_DT,
        region=set_conn1
    )

    # 8. 载荷与边界条件：Smooth Step 升速 + 从动轮等效阻力矩
    m.SmoothStepAmplitude(
        name='Amp-Smooth',
        timeSpan=STEP,
        data=((0.0, 0.0), (RAMP_TIME, 1.0), (STEP_TIME, 1.0))
    )
    m.VelocityBC(
        name='BC-Speed',
        createStepName='Step-1',
        region=set_rp1,
        v1=UNSET, v2=UNSET, v3=UNSET,
        vr1=UNSET, vr2=UNSET, vr3=INPUT_OMEGA,
        amplitude='Amp-Smooth',
        distributionType=UNIFORM
    )
    m.Moment(
        name='Load-Torque',
        createStepName='Step-1',
        region=set_rp2,
        cm3=LOAD_TORQUE,
        amplitude='Amp-Smooth',
        distributionType=UNIFORM
    )

    a.regenerate()
    return {
        'model': model_name,
        'gear29_elems': len(p29.elements),
        'gear95_elems': len(p95.elements),
        'total_elems': len(p29.elements) + len(p95.elements),
    }


def build_all():
    if not os.path.isdir(WORKDIR):
        os.makedirs(WORKDIR)
    os.chdir(WORKDIR)

    # 清空旧作业，重建标准双模型工程
    for jname in list(mdb.jobs.keys()):
        del mdb.jobs[jname]

    info_h = build_single_model('Model-Healthy', missing_tooth_idx=-1)
    info_m = build_single_model('Model-Missing', missing_tooth_idx=15)

    # 删除残留的旧默认模型
    for old_name in list(mdb.models.keys()):
        if old_name not in ('Model-Healthy', 'Model-Missing'):
            del mdb.models[old_name]

    # 创建两套作业（8 核 + 双精度分析打包）
    for jname, mname, desc in [
        ('Job-Healthy', 'Model-Healthy', 'Stage 1 gear pair - Healthy baseline (0.5 s, 30 Hz)'),
        ('Job-Missing', 'Model-Missing', 'Stage 1 gear pair - SS1.5-95 tooth #15 missing (0.5 s, 30 Hz)'),
    ]:
        mdb.Job(
            name=jname,
            model=mname,
            description=desc,
            type=ANALYSIS,
            numCpus=NUM_CPUS,
            numDomains=NUM_CPUS,
            multiprocessingMode=DEFAULT,
            explicitPrecision=DOUBLE_PLUS_PACK,
            nodalOutputPrecision=SINGLE,
        )

    cae_path = os.path.join(WORKDIR, CAE_FILENAME)
    mdb.saveAs(pathName=cae_path)
    return {
        'cae_path': cae_path,
        'healthy': info_h,
        'missing': info_m,
        'jobs': list(mdb.jobs.keys()),
    }


if __name__ == '__main__':
    res = build_all()
    print('[DONE] Stage-1 gearbox models built:', res)
