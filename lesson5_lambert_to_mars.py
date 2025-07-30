# lesson5_lambert_to_mars.py (SỬA LỖI PLOTTING CUỐI CÙNG)

from astropy.time import Time
import astropy.units as u
import numpy as np
import sys

from poliastro.bodies import Sun, Earth, Mars
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit
from poliastro.iod import lambert
from poliastro.plotting import OrbitPlotter3D

print("Poliastro - Bài học 5: Bài toán Lambert - Hành trình đến Sao Hỏa")

# --- Bước 1 & 2: Giữ nguyên ---
launch_date = Time("1964-11-28 14:22", scale="utc")
arrival_date = Time("1965-07-15 01:00", scale="utc")
time_of_flight = arrival_date - launch_date

print(f"\nKhởi hành từ Trái Đất: {launch_date.iso}")
print(f"Dự kiến đến Sao Hỏa: {arrival_date.iso}")
print(f"Thời gian bay dự kiến: {time_of_flight.to(u.day):.2f}")

earth_ephem = Ephem.from_body(Earth, launch_date)
mars_ephem = Ephem.from_body(Mars, arrival_date)

r1_raw, v_earth_raw = earth_ephem.rv()
r2_raw, v_mars_raw = mars_ephem.rv()

r1 = r1_raw.flatten()
v_earth = v_earth_raw.flatten()
r2 = r2_raw.flatten()
v_mars = v_mars_raw.flatten()

# --- Bước 3: Giữ nguyên ---
print("\nĐang giải bài toán Lambert...")
try:
    v1, v2 = lambert(Sun.k, r1, r2, time_of_flight, prograde=True)
except Exception as e:
    print(f"\n!!! HÀM LAMBERT GẶP LỖI: {e} !!!")
    sys.exit()

# --- Bước 4: Tạo quỹ đạo và vẽ (Sửa cách vẽ điểm) ---
print("\n--- Kết quả từ bài toán Lambert ---")
print(f"Vận tốc Trái Đất (so với Mặt Trời): {v_earth.to(u.km/u.s)}")
print(f"Vận tốc cần có để phóng (so với Mặt Trời): {v1.to(u.km/u.s)}")
print(f"Vận tốc khi đến Sao Hỏa (so với Mặt Trời): {v2.to(u.km/u.s)}")
print(f"Vận tốc Sao Hỏa (so với Mặt Trời): {v_mars.to(u.km/u.s)}")

mars_transfer_orbit = Orbit.from_vectors(Sun, r1, v1, epoch=launch_date)
earth_orbit = Orbit.from_vectors(Sun, r1, v_earth, epoch=launch_date)
mars_orbit = Orbit.from_vectors(Sun, r2, v_mars, epoch=arrival_date)

print("\nĐang tạo biểu đồ 3D cho hành trình liên hành tinh...")
plotter = OrbitPlotter3D()
plotter.plot(earth_orbit, label="Quỹ đạo Trái Đất")
plotter.plot(mars_orbit, label="Quỹ đạo Sao Hỏa")
plotter.plot(mars_transfer_orbit, label="Quỹ đạo chuyển tiếp đến Sao Hỏa", color="red")

# THAY ĐỔI Ở ĐÂY: Dùng lại .plot() để vẽ các điểm.
# Hàm .plot() sẽ tự động vẽ một marker nếu nó nhận ra đó là một điểm.
# Chúng ta vẽ chính các đối tượng quỹ đạo tại điểm đó.
plotter.plot(earth_orbit, label="Trái Đất (khởi hành)", color="blue")
plotter.plot(mars_orbit, label="Sao Hỏa (đến)", color="orange")


fig = plotter._figure
output_filename = "lambert_to_mars.png"
fig.update_layout(width=1200, height=900, title_text="Hành trình đến Sao Hỏa (Bài toán Lambert)")
fig.write_image(output_filename)

print(f"-> Đã lưu biểu đồ thành công vào file: {output_filename}")
