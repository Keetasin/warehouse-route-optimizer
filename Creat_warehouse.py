import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# =========================================================
# 1) กำหนดแผนที่โกดัง
#    0 = ทางเดิน
#    1 = ชั้นวางสินค้า / สิ่งกีดขวาง
# =========================================================
warehouse = np.array([
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]])

# =========================================================
# 2) การเคลื่อนที่แบบ Manhattan: บน ล่าง ซ้าย ขวา
# =========================================================
movements = [
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1)
]
# =========================================================
# 3) ตรวจสอบว่าพิกัดอยู่ภายในแผนที่หรือไม่
# =========================================================
def is_valid_position(warehouse, position):
    row, col = position
    rows, cols = warehouse.shape
    return 0 <= row < rows and 0 <= col < cols
# =========================================================
# 4) ตรวจสอบประเภทของจุด
# =========================================================
def validate_points(warehouse, entrance, exit_point, pickup_points):
    # ทางเข้าและทางออกต้องอยู่บนทางเดิน
    if warehouse[entrance] != 0:
        raise ValueError("Entrance ต้องอยู่บนทางเดินที่มีค่า 0")

    if warehouse[exit_point] != 0:
        raise ValueError("Exit ต้องอยู่บนทางเดินที่มีค่า 0")

    # จุดหยิบสินค้าทุกจุดต้องอยู่บนชั้นวาง
    for point in pickup_points:
        if warehouse[point] != 1:
            raise ValueError(
                f"Pickup point {point} ต้องอยู่บนชั้นวางที่มีค่า 1"
            )

# =========================================================
# 5) สุ่มจุดหยิบสินค้าจากตำแหน่งชั้นวาง
# =========================================================
def generate_pickup_points(warehouse, n, seed=42):
    # หาพิกัดทุกจุดที่เป็นชั้นวาง (ค่า 1)
    shelf_positions = list(zip(*np.where(warehouse == 1)))
    if n > len(shelf_positions):
        raise ValueError(
            f"ไม่สามารถสร้างจุดหยิบสินค้า {n} จุดได้ "
            f"เพราะมีชั้นวางทั้งหมดเพียง {len(shelf_positions)} จุด"
        )
    # ตั้งค่า seed เพื่อให้ผลการสุ่มเหมือนเดิมทุกครั้ง
    rng = np.random.default_rng(seed)

    # เลือกตำแหน่งชั้นวางแบบไม่ซ้ำกัน
    selected_indices = rng.choice(
        len(shelf_positions),
        size=n,
        replace=False
    )

    pickup_points = [
        shelf_positions[index]
        for index in selected_indices
    ]
    return pickup_points


def generate_pickup_points2(warehouse, n):
    # หาพิกัดทุกจุดที่เป็นชั้นวาง (ค่า 1)
    shelf_positions = list(zip(*np.where(warehouse == 1)))

    if n > len(shelf_positions):
        raise ValueError(
            f"ไม่สามารถสร้างจุดหยิบสินค้า {n} จุดได้ "
            f"เพราะมีชั้นวางทั้งหมดเพียง {len(shelf_positions)} จุด"
        )

    # ใช้เวลาปัจจุบันเป็น seed
    seed = int(time.time())

    # สร้าง random generator
    rng = np.random.default_rng(seed)

    # เลือกตำแหน่งชั้นวางแบบไม่ซ้ำกัน
    selected_indices = rng.choice(
        len(shelf_positions),
        size=n,
        replace=False
    )

    pickup_points = [
        shelf_positions[index]
        for index in selected_indices
    ]

    return pickup_points
# =========================================================
# 6) แสดงแผนที่ พร้อมทางเข้า ทางออก และจุดหยิบสินค้า
# =========================================================
def plot_warehouse(warehouse, entrance, exit_point, pickup_points):
    rows, cols = warehouse.shape
    # 0 = ทางเดิน, 1 = ชั้นวาง
    colors = [
        "#EAF6FF",  # ฟ้าอ่อน = ทางเดิน
        "#4A4A4A"   # เทาเข้ม = ชั้นวาง
    ]

    warehouse_cmap = ListedColormap(colors)
    plt.figure(figsize=(15, 10))
    plt.imshow(
        warehouse,
        cmap=warehouse_cmap,
        origin="upper",
        interpolation="nearest"
    )
    # แสดงจุดทางเข้า
    plt.scatter(
        entrance[1],
        entrance[0],
        color="#27AE60",
        s=220,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        zorder=3,
        label="Entrance"
    )

    plt.text(
        entrance[1],
        entrance[0],
        "IN",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="white",
        zorder=4
    )
    # แสดงจุดทางออก
    plt.scatter(
        exit_point[1],
        exit_point[0],
        color="#E74C3C",
        s=220,
        marker="o",
        edgecolors="black",
        linewidths=1.2,
        zorder=3,
        label="Exit"

    )

    plt.text(
        exit_point[1],
        exit_point[0],
        "OUT",
        ha="center",
        va="center",
        fontsize=7,
        fontweight="bold",
        color="white",
        zorder=4

    )
   # แสดงจุดหยิบสินค้า

    for index, point in enumerate(pickup_points, start=1):
        plt.scatter(
            point[1],
            point[0],
            color="#F1C40F",
            s=180,
            marker="s",
            edgecolors="black",
            linewidths=1.0,
            zorder=3
        )

        plt.text(
            point[1],
            point[0],
            str(index),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="black",
            zorder=4

        )


    # ตั้งค่าแกนและเส้นตาราง
    plt.xticks(np.arange(cols))
    plt.yticks(np.arange(rows))

    plt.xticks(np.arange(-0.5, cols, 1), minor=True)
    plt.yticks(np.arange(-0.5, rows, 1), minor=True)

    plt.grid(
        which="minor",
        color="white",
        linestyle="-",
        linewidth=0.8
    )

    plt.tick_params(which="minor", bottom=False, left=False)
    plt.title(
        "Warehouse Map with Entrance, Exit and Pickup Points",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("Column")
    plt.ylabel("Row")
    # สร้างคำอธิบายสัญลักษณ์

    legend_elements = [
        Patch(facecolor="#EAF6FF", edgecolor="black", label="Walkway (0)"),
        Patch(facecolor="#4A4A4A", edgecolor="black", label="Shelf (1)"),
        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Entrance",
            markerfacecolor="#27AE60",
            markeredgecolor="black",
            markersize=10

        ),

        plt.Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="Exit",
            markerfacecolor="#E74C3C",
            markeredgecolor="black",
            markersize=10
        ),

        plt.Line2D(
            [0], [0],
            marker="s",
            color="w",
            label="Pickup Point",
            markerfacecolor="#F1C40F",
            markeredgecolor="black",
            markersize=10

        )
    ]

    plt.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=5
    )

    plt.tight_layout()
    plt.show()

# =========================================================
# 7) กำหนดทางเข้า ทางออก และจำนวนจุดหยิบสินค้า
# =========================================================
# พิกัดอยู่ในรูปแบบ (row, column)
entrance = (0, 0)      # ต้องเป็นทางเดิน: ค่า 0
exit_point = (17, 19)  # ต้องเป็นทางเดิน: ค่า 0
# จำนวนจุดที่ต้องหยิบสินค้า
n = 10  
# สุ่มจุดหยิบสินค้าบนชั้นวาง: ทุกจุดจะมีค่า 1 แน่นอน
pickup_points = generate_pickup_points( warehouse=warehouse, n=n, seed=20)


#pickup_points = generate_pickup_points2( warehouse=warehouse, n=n)
# ตรวจสอบความถูกต้องของจุดทั้งหมด
validate_points(
    warehouse=warehouse,
    entrance=entrance,
    exit_point=exit_point,
    pickup_points=pickup_points

)

# =========================================================
# 8) แสดงข้อมูลพิกัด
# =========================================================

print("Entrance:", entrance)
print("Exit:", exit_point)
print(f"Pickup points ({n} points):")

for index, point in enumerate(pickup_points, start=1):
    print(f"  Pickup {index}: {point}")

# =========================================================
# 9) แสดงแผนที่
# =========================================================

plot_warehouse(
    warehouse=warehouse,
    entrance=entrance,
    exit_point=exit_point,
    pickup_points=pickup_points

)