# lesson1_basic_orbit.py (PHIÊN BẢN PLOTLY-NATIVE)

# --- Các thư viện cần thiết ---
import astropy.units as u
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting import OrbitPlotter2D

print("Chào mừng đến với Poliastro - Bài học 1: Quỹ đạo cơ bản")

# --- Bước 1, 2, 3 giữ nguyên ---

# --- Bước 1: Định nghĩa các tham số của quỹ đạo ---
a = 24396.15 * u.km
ecc = 0.7308 * u.one
inc = 7.0 * u.deg
raan = 180.0 * u.deg
argp = 0.0 * u.deg
nu = 0.0 * u.deg

# --- Bước 2: Tạo đối tượng Orbit ---
gto_orbit = Orbit.from_classical(
    attractor=Earth,
    a=a,
    ecc=ecc,
    inc=inc,
    raan=raan,
    argp=argp,
    nu=nu
)

# --- Bước 3: In thông tin về quỹ đạo ---
print("\n--- Thông tin quỹ đạo ---")
print(gto_orbit)
perigee_altitude = gto_orbit.r_p - Earth.R
apogee_altitude = gto_orbit.r_a - Earth.R
print(f"Độ cao cận điểm (perigee): {perigee_altitude.to(u.km):.2f}")
print(f"Độ cao viễn điểm (apogee): {apogee_altitude.to(u.km):.2f}")
print(f"Chu kỳ quỹ đạo: {gto_orbit.period.to(u.hour):.2f}")

# --- Bước 4: Vẽ và LƯU quỹ đạo (Cách làm cho Plotly) ---
print("\nĐang tạo biểu đồ 2D...")

# 1. Tạo plotter. Nó sẽ tự động dùng Plotly backend.
plotter = OrbitPlotter2D()
plotter.plot(gto_orbit, label="Quỹ đạo GTO")

# 2. Lấy đối tượng figure của Plotly từ thuộc tính _figure
fig = plotter._figure

# 3. Dùng phương thức write_image() để lưu file. ĐÂY LÀ ĐIỂM MẤU CHỐT.
output_filename = "gto_orbit_2d.png"
fig.write_image(output_filename)

print(f"-> Đã lưu biểu đồ thành công vào file: {output_filename}")
