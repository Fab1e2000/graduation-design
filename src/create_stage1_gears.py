# -*- coding: utf-8 -*-
"""
Abaqus 2024 建模辅助脚本：一键生成第一级齿轮副精确 2D 渐开线部件
包含：
  1. Gear_29          (SS1.5-29 主动轮，带内孔与齿根网格加密分区圆)
  2. Gear_95_Healthy  (SS1.5-95 从动轮健康版，已预设啮合相位，平移至 x=93 即可咬合)
  3. Gear_95_Missing  (SS1.5-95 从动轮单齿完全缺失版，缺失第 15 号齿)

使用方法（Abaqus/CAE 2024 中文版）：
  菜单栏 -> 文件(File) -> 运行脚本(Run Script) -> 选择本文件即可。
单位制：mm（长度）
"""
import math
from abaqus import mdb
from abaqusConstants import TWO_D_PLANAR, DEFORMABLE_BODY


def involute_gear_points(Z, m=1.5, alpha_deg=20.0, angle_offset=0.0,
                         missing_tooth_idx=-1, n_flank=4):
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

    # 齿槽半角（根圆处）
    r_start = max(rb, rf)
    th_start = inv_theta(r_start)
    # 齿根圆角过渡处取略大于 th_start
    th_root = min(th_start + 0.08 * step, 0.42 * step)

    pts = []
    for i in range(Z):
        a0 = i * step + angle_offset

        if i == missing_tooth_idx:
            # 单齿完全缺失：沿齿根圆直接圆弧过渡
            for frac in (-0.42, -0.21, 0.0, 0.21, 0.42):
                ang = a0 + frac * step
                pts.append((rf * math.cos(ang), rf * math.sin(ang)))
            continue

        # 左侧齿根过渡点
        a_lr = a0 - th_root
        pts.append((rf * math.cos(a_lr), rf * math.sin(a_lr)))

        # 左侧渐开线（由根到顶）
        for k in range(n_flank + 1):
            rk = r_start + (ra - r_start) * k / float(n_flank)
            ang = a0 - inv_theta(rk)
            pts.append((rk * math.cos(ang), rk * math.sin(ang)))

        # 齿顶中点
        pts.append((ra * math.cos(a0), ra * math.sin(a0)))

        # 右侧渐开线（由顶到根）
        for k in range(n_flank, -1, -1):
            rk = r_start + (ra - r_start) * k / float(n_flank)
            ang = a0 + inv_theta(rk)
            pts.append((rk * math.cos(ang), rk * math.sin(ang)))

        # 右侧齿根过渡点
        a_rr = a0 + th_root
        pts.append((rf * math.cos(a_rr), rf * math.sin(a_rr)))

    return pts, rf


def create_gear_part(model, part_name, Z, m=1.5, r_bore=12.7,
                     angle_offset=0.0, missing_tooth_idx=-1):
    """在指定模型中创建 2D 平面齿轮壳体部件并划分齿根加密分区圆。"""
    pts, rf = involute_gear_points(
        Z=Z, m=m, angle_offset=angle_offset, missing_tooth_idx=missing_tooth_idx
    )

    sheet_size = max(200.0, m * Z * 3.0)
    s = model.ConstrainedSketch(name='__profile_' + part_name, sheetSize=sheet_size)

    # 1. 绘制外齿廓闭环折线
    n = len(pts)
    for i in range(n):
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) > 1e-6:
            s.Line(point1=p1, point2=p2)

    # 2. 绘制中心轴孔圆 (ER-16K 1英寸轴径 -> r = 12.7 mm)
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r_bore, 0.0))

    # 3. 生成 2D 平面可变形壳体
    p = model.Part(
        name=part_name,
        dimensionality=TWO_D_PLANAR,
        type=DEFORMABLE_BODY
    )
    p.BaseShell(sketch=s)
    del model.sketches['__profile_' + part_name]

    # 4. 创建网格加密分区圆（把齿圈外环与内部轮辐分开，方便局部加密）
    r_part = rf - (4.0 if Z < 50 else 8.5)
    if r_part > r_bore + 2.0:
        sp = model.ConstrainedSketch(name='__part_' + part_name, sheetSize=sheet_size)
        sp.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(r_part, 0.0))
        p.PartitionFaceBySketch(faces=p.faces[:], sketch=sp)
        del model.sketches['__part_' + part_name]

    print('[OK] 已生成部件: %s (Z=%d, m=%.1f, 缺齿索引=%d)' % (part_name, Z, m, missing_tooth_idx))
    return p


def main():
    model_name = 'Model-1'
    if model_name not in mdb.models:
        mdb.Model(name=model_name)
    model = mdb.models[model_name]

    # Gear_29: 齿 #0 指向 +X 方向
    create_gear_part(
        model, 'Gear_29', Z=29, m=1.5, r_bore=12.7,
        angle_offset=0.0, missing_tooth_idx=-1
    )

    # Gear_95: 旋转 pi + pi/95，使齿槽正对 -X；装配时直接平移 (93.0, 0, 0) 即可精确啮合
    offset_95 = math.pi + math.pi / 95.0
    create_gear_part(
        model, 'Gear_95_Healthy', Z=95, m=1.5, r_bore=12.7,
        angle_offset=offset_95, missing_tooth_idx=-1
    )

    # 缺失第 15 号齿（避开 t=0 初始接触点，确保启动平滑升速后再进入缺齿冲击）
    create_gear_part(
        model, 'Gear_95_Missing', Z=95, m=1.5, r_bore=12.7,
        angle_offset=offset_95, missing_tooth_idx=15
    )

    print('=== 全部 3 个第一级齿轮部件创建完成！请继续按操作指南进行属性与装配设置。===')


if __name__ == '__main__':
    main()
