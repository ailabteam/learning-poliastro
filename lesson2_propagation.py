# lesson2_propagation.py

# --- Các thư viện cần thiết ---
import astropy.units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter3D

print("Poliastro - Bài học 2: Truyền bá quỹ đạo và Vẽ 3D")

# --- Bước 1: Tạo một quỹ đạo đã biết ---
# Lần này, chúng ta sẽ tạo quỹ đạo của Trạm vũ trụ Quốc tế (ISS)
# bằng vector vị trí và vận tốc (State Vectors).
# Dữ liệu này là gần đúng tại một thời điểm nhất định.
# Định nghĩa thời điểm tham chiếu (epoch)
epoch = Time("2024-01-01 12:00:00", scale="utc")

# Vector vị trí (r) và vận tốc (v)
r = [-2384.46, 5729.01, 3050.46] * u.km
v = [-7.37, -2.98, 1.64] * u.km / u.s

# Tạo quỹ đạo từ vector trạng thái
iss_orbit = Orbit.from_vectors(Earth, r, v, epoch=epoch)

print("\n--- Quỹ đạo ISS ban đầu ---")
print(iss_orbit)

# --- Bước 2: Truyền bá quỹ đạo (Propagation) ---
# Chúng ta muốn biết ISS sẽ ở đâu sau 45 phút.
dt = 45 * u.min
print(f"\nĐang truyền bá quỹ đạo đi {dt}...")

iss_orbit_future = iss_orbit.propagate(dt)

print("\n--- Quỹ đạo ISS sau 45 phút ---")
print(iss_orbit_future)

# Bạn có thể thấy các tham số quỹ đạo hầu như không đổi (vì nó vẫn trên cùng một quỹ đạo),
# nhưng vị trí của nó (được biểu thị bằng dị thường thực `nu`) đã thay đổi.
print(f"Dị thường thực ban đầu: {iss_orbit.nu:.2f}")
print(f"Dị thường thực sau 45 phút: {iss_orbit_future.nu:.2f}")


# --- Bước 3: Vẽ và LƯU quỹ đạo 3D ---
# Vẽ 3D phức tạp hơn một chút, nhưng trông rất chuyên nghiệp.
# OrbitPlotter3D là công cụ cho việc này.
print("\nĐang tạo biểu đồ 3D...")
plotter = OrbitPlotter3D()

# Vẽ quỹ đạo đầy đủ
plotter.plot(iss_orbit, label=f"ISS Quỹ đạo (Epoch: {epoch.iso})")

# Để làm nổi bật sự thay đổi vị trí, chúng ta có thể vẽ các điểm riêng biệt.
# Hàm scatter vẽ các điểm thay vì cả quỹ đạo.
plotter.scatter(iss_orbit, label="Vị trí ban đầu", color="blue", s=100) # s là kích thước điểm
plotter.scatter(iss_orbit_future, label=f"Vị trí sau {dt}", color="red", s=100)

# Lấy đối tượng figure và lưu lại
fig = plotter.get_figure()
output_filename = "iss_propagation_3d.png"

# Với biểu đồ 3D, chúng ta nên đặt kích thước lớn hơn để nhìn rõ
fig.set_size_inches(10, 8)
fig.savefig(output_filename, dpi=150) # dpi tăng độ phân giải

print(f"-> Đã lưu biểu đồ 3D thành công vào file: {output_filename}")
