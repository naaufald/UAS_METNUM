import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

st.markdown(
    """
    <style>
        .stApp {
            background-color: white !important;
        }
        body, .stMarkdown, .stTextInput, .stButton, .stSelectbox, .stSlider, .stDataFrame, .stTable {
            color: black !important;
        }
        h1, h2, h3, h4, h5, h6, .stHeader, .stSubheader {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Differential Equation: Runge-Kutta Method for Simulation")
st.header("Flowchart")

st.image(r"D:/Matana/Semester 4/Metode Numerik/UAS/Flowchart.jpg", caption="Diagram Flowchart")


st.header("Runge-Kutta Method for Temperature Simulation")

# Section: Newton's Law of Cooling
st.header("Newton's Law of Cooling - Runge-Kutta Equation")

st.markdown(r"""
### 🧊 Newton's Law of Cooling

Persamaan diferensial untuk hukum pendinginan Newton:
$$
\frac{dT}{dt} = -k(T - T_{\text{env}})
$$

di mana:
- \( T \) = suhu objek saat ini,
- \( T env \) = suhu lingkungan (konstan),
- \( k \) = konstanta pendinginan (positif).

### Runge-Kutta (RK4) Method untuk Simulasi:
Langkah-langkah metode Runge-Kutta orde 4 (RK4) untuk menyelesaikan:
$$
\frac{dT}{dt} = f(t, T)
$$

adalah:

```python
def f(t, T):
    return -k * (T - T_env)

for i in range(n):
    k1 = h * f(t, T)
    k2 = h * f(t + h/2, T + k1/2)
    k3 = h * f(t + h/2, T + k2/2)
    k4 = h * f(t + h, T + k3)
    
    T = T + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
    t = t + h""")


st.title("Simulasi Pendinginan Menggunakan Runge-Kutta (Data Suhu Jakarta)")

st.header("Data : City_Temperature")
st.markdown("""data ini mencakup suhu harian yang ada di berbagai negara dengan variabel sebagai berikut:
- region  
- country  
- state  
- city  
- month  
- day  
- year  
- average temperature""")
uploaded_file = st.file_uploader("Upload dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())
    st.write("**Sumber Data:** [Kaggle.com](https://www.kaggle.com/datasets/sudalairajkumar/daily-temperature-of-major-cities)")

# Filter data Jakarta
jakarta_data = df[df['City'] == 'Jakarta'].copy()
jakarta_data['Date'] = pd.to_datetime(jakarta_data[['Year', 'Month', 'Day']])
jakarta_data.set_index('Date', inplace=True)

selected_month = st.selectbox("Pilih Bulan", sorted(jakarta_data.index.month.unique()))
selected_year = st.selectbox("Pilih Tahun", sorted(jakarta_data.index.year.unique()))
monthly_data = jakarta_data[(jakarta_data.index.month == selected_month) & 
                                (jakarta_data.index.year == selected_year)]

monthly_data = monthly_data[monthly_data['AvgTemperature'] > -50]
## Inisialisasi parameter simulasi
T0 = monthly_data['AvgTemperature'].iloc[0]
T_env = monthly_data['AvgTemperature'].mean()
k = st.slider("Konstanta Pendinginan (k)", 0.01, 1.0, 0.1)
dt = 1
steps = len(monthly_data) - 1

def runge_kutta(T0, T_env, k, dt, steps):
        T = [T0]
        for _ in range(steps):
            T_curr = T[-1]
            k1 = -k * (T_curr - T_env)
            k2 = -k * ((T_curr + 0.5 * dt * k1) - T_env)
            k3 = -k * ((T_curr + 0.5 * dt * k2) - T_env)
            k4 = -k * ((T_curr + dt * k3) - T_env)
            T_next = T_curr + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            T.append(T_next)
        return T

simulated = runge_kutta(T0, T_env, k, dt, steps)

st.markdown("""
```python
jakarta_data = df[df['City'] == 'Jakarta'].copy()
jakarta_data['Date'] = pd.to_datetime(jakarta_data[['Year', 'Month', 'Day']])
jakarta_data.set_index('Date', inplace=True)

selected_month = st.selectbox("Pilih Bulan", sorted(jakarta_data.index.month.unique()))
selected_year = st.selectbox("Pilih Tahun", sorted(jakarta_data.index.year.unique()))
monthly_data = jakarta_data[(jakarta_data.index.month == selected_month) & 
                                (jakarta_data.index.year == selected_year)]""")

# Buat DataFrame hasil
simulasi_df = pd.DataFrame({
        "Tanggal": monthly_data.index[:len(simulated)],
        "Suhu Aktual": monthly_data['AvgTemperature'].values[:len(simulated)],
        "Suhu Simulasi (RK4)": simulated
    }).set_index("Tanggal")
 # Tampilkan tabel hasil simulasi
st.subheader("Tabel Hasil Simulasi")
st.dataframe(simulasi_df)

st.subheader("Visualisasi Suhu")
fig_line = px.line(simulasi_df, y=["Suhu Aktual", "Suhu Simulasi (RK4)"],
                       title="Suhu Aktual vs Simulasi (Runge-Kutta)")
st.plotly_chart(fig_line)

st.markdown("data yang muncul pada tabel merupakan data yang telah diseleksi, yaitu menggunakan data daerah jakarta.")

st.markdown("""
```python
simulasi_df = pd.DataFrame({
    "Tanggal": monthly_data.index[:len(simulated)],
    "Suhu Aktual": monthly_data['AvgTemperature'].values[:len(simulated)],
    "Suhu Simulasi (RK4)": simulated
})
simulasi_df.set_index("Tanggal", inplace=True)

# Tampilkan tabel hasil simulasi
st.subheader("Tabel Hasil Simulasi")
st.dataframe(simulasi_df)

st.subheader("Visualisasi Suhu")
fig_line = px.line(simulasi_df, y=["Suhu Aktual", "Suhu Simulasi (RK4)"],
                   title="Suhu Aktual vs Simulasi (Runge-Kutta)")
st.plotly_chart(fig_line)""")

st.subheader("Visualisasi Suhu")
fig_line = px.line(simulasi_df, y=["Suhu Aktual", "Suhu Simulasi (RK4)"],
                   title="Suhu Aktual vs Simulasi (Runge-Kutta)")
# Add a unique key to the plotly_chart
st.plotly_chart(fig_line, key="unique_chart_key_1")

st.markdown("""
```python
st.subheader("Visualisasi Suhu")
fig_line = px.line(simulasi_df, y=["Suhu Aktual", "Suhu Simulasi (RK4)"],
                   title="Suhu Aktual vs Simulasi (Runge-Kutta)")
st.plotly_chart(fig_line)""")

st.subheader("Statistik Deskriptif")
st.write(simulasi_df.describe())

st.markdown("""
```python
st.write(simulasi_df.describe())""")
st.markdown("berdasarkan deskriptif data, didapatkan bahwa suhu di jakarta memiliki rata-rata 81,14 (suhu aktual, menggunakan fahrenheit)," \
"dengan suhu simulasi runge-kutta sekitar 82,54 (menggunakan fahrenheit). terdapat error sekitar 1,3998 dari data simulasi dan juga data aktual.")

###runge-kutta for motion simulation

st.header("Runge-Kutta Method for Motion Simulation (Mechanics)")

st.markdown("### Persamaan Dasar")
st.latex(r"F = m \cdot a \Rightarrow a = \frac{F}{m}")
st.latex(r"\frac{dv}{dt} = a, \quad \frac{dx}{dt} = v")

st.markdown("""
Metode Runge-Kutta orde 4 digunakan untuk menyelesaikan sistem diferensial ini secara numerik.  
Aplikasi dari simulasi ini meliputi:
- Prediksi posisi dan kecepatan benda di bawah gaya konstan atau variabel
- Analisis gerak peluru
- Simulasi gerak harmonik, dll.
""")

st.markdown("### Kode Simulasi (Gaya konstan)")
st.code('''
def runge_kutta_motion(m, F, x0, v0, t0, t_end, h):
    def acceleration(t, x, v):
        return F / m  # Gaya konstan

    times = [t0]
    positions = [x0]
    velocities = [v0]

    t = t0
    x = x0
    v = v0

    while t < t_end:
        k1v = h * acceleration(t, x, v)
        k1x = h * v

        k2v = h * acceleration(t + h/2, x + k1x/2, v + k1v/2)
        k2x = h * (v + k1v/2)

        k3v = h * acceleration(t + h/2, x + k2x/2, v + k2v/2)
        k3x = h * (v + k2v/2)

        k4v = h * acceleration(t + h, x + k3x, v + k3v)
        k4x = h * (v + k3v)

        x += (k1x + 2*k2x + 2*k3x + k4x) / 6
        v += (k1v + 2*k2v + 2*k3v + k4v) / 6
        t += h

        times.append(t)
        positions.append(x)
        velocities.append(v)

    return pd.DataFrame({'Time': times, 'Position': positions, 'Velocity': velocities})
''', language='python')

def runge_kutta_motion(m, F, x0, v0, t0, t_end, h):
    def acceleration(t, x, v):
        return F / m  # Gaya konstan

    times = [t0]
    positions = [x0]
    velocities = [v0]

    t = t0
    x = x0
    v = v0

    while t < t_end:
        k1v = h * acceleration(t, x, v)
        k1x = h * v

        k2v = h * acceleration(t + h/2, x + k1x/2, v + k1v/2)
        k2x = h * (v + k1v/2)

        k3v = h * acceleration(t + h/2, x + k2x/2, v + k2v/2)
        k3x = h * (v + k2v/2)

        k4v = h * acceleration(t + h, x + k3x, v + k3v)
        k4x = h * (v + k3v)

        x += (k1x + 2*k2x + 2*k3x + k4x) / 6
        v += (k1v + 2*k2v + 2*k3v + k4v) / 6
        t += h

        times.append(t)
        positions.append(x)
        velocities.append(v)

    return pd.DataFrame({'Time': times, 'Position': positions, 'Velocity': velocities})
                        
# Simulasi
m, F, x0, v0, t0, t_end, h = 1.0, 10.0, 0.0, 0.0, 0.0, 10.0, 0.1
df = runge_kutta_motion(m, F, x0, v0, t0, t_end, h)

st.markdown("### Data Hasil Simulasi")
st.dataframe(df)
st.markdown("Data simulasi merupakan hasil dari penerapan metode numerik Runge-Kutta orde 4 untuk menyelesaikan hukum gerak Newton kedua dengan F = 10N dan m = 1kg, sehingga didapatkan a = 10 m/s^2. dimana, runge-kutta digunakan dalam menghitung perubahan kecepatan dan posisi secara bertahap berdasarkan waktu. sehingga, pada data hasil simulasi menunjukkan adanya kecepatan yang meningkat secara linear karena percepatan yang konstan dan posisi yang meningkat secara kuadratik karena kecepatan yang bertambah seiring waktu")

st.markdown("### Visualisasi Gerak")
fig1 = px.line(df, x='Time', y='Position', title='Posisi vs Waktu')
fig2 = px.line(df, x='Time', y='Velocity', title='Kecepatan vs Waktu')
st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.markdown("### Analisis Distribusi Data")
st.markdown("Distribusi posisi dan kecepatan sepanjang waktu:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Histogram Posisi")
    st.plotly_chart(px.histogram(df, x='Position', nbins=20))
with col2:
    st.markdown("#### Histogram Kecepatan")
    st.plotly_chart(px.histogram(df, x='Velocity', nbins=20))


st.header("Runge-Kutta Method for Simulating Fluid Flow or Heat Conduction Over Time")

# --- Penjelasan Persamaan ---
st.subheader(" Runge-Kutta Equation for Heat Transfer and Fluid Dynamics (CFD)")
st.markdown(r"""
Persamaan umum **Runge-Kutta orde 4 (RK4)**:
$$
\begin{aligned}
k_1 &= f(t_n, y_n) \\
k_2 &= f\left(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1\right) \\
k_3 &= f\left(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_2\right) \\
k_4 &= f\left(t_n + h, y_n + hk_3\right) \\
y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
$$

Contoh aplikasi:
- **Heat conduction**: 1D heat
- **Fluid dynamics**: Euler’s equation atau Navier-Stokes (disederhanakan menjadi ODE untuk RK)

""")

# --- Kode Simulasi: Heat Conduction Sederhana ---
st.subheader(" Simulation Code (1D Heat Equation)")
st.markdown("Berikut adalah implementasi RK4 untuk simulasi suhu sepanjang batang (1D):")

code = '''
import numpy as np

def heat_rhs(T, alpha, dx):
    d2T_dx2 = np.zeros_like(T)
    d2T_dx2[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2]) / dx**2
    return alpha * d2T_dx2

def rk4_step(T, dt, dx, alpha):
    k1 = heat_rhs(T, alpha, dx)
    k2 = heat_rhs(T + 0.5 * dt * k1, alpha, dx)
    k3 = heat_rhs(T + 0.5 * dt * k2, alpha, dx)
    k4 = heat_rhs(T + dt * k3, alpha, dx)
    return T + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
'''

st.code(code, language="python")

# --- Simulasi dan Data ---
st.subheader(" Simulasi Data Hasil")

# Parameter simulasi
L = 1.0
nx = 50
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

alpha = 0.01
dt = 0.001
nt = 100

# Inisialisasi suhu
T = np.zeros(nx)
T[int(nx/4):int(nx/2)] = 100  # suhu awal di tengah batang

# Simpan hasil simulasi
results = [T.copy()]
for n in range(nt):
    T = T + dt * (alpha * (np.roll(T, -1) - 2 * T + np.roll(T, 1)) / dx**2)
    T[0] = T[-1] = 0  # boundary condition
    results.append(T.copy())

results = np.array(results)

df = pd.DataFrame(results, columns=[f"x={round(xi, 2)}" for xi in x])
st.dataframe(df.head())

st.markdown("""
Berikut merupakan data hasil simulasi penyebaran suhu yang diselesaikan menggunakan metode numerik.  
Baris pada tabel menunjukkan distribusi suhu pada waktu tertentu, sedangkan kolom merepresentasikan posisi tertentu pada batang.
Berdasarkan data tersebut, dapat dilihat bahwa suhu awal yang diberikan di tengah batang mulai menyebar ke seluruh panjang batang seiring berjalannya waktu.
""")


# --- Visualisasi ---
st.subheader(" Visualisasi: Suhu terhadap waktu")

fig, ax = plt.subplots(figsize=(10, 4))
for t in [0, 25, 50, 75, 99]:
    ax.plot(x, results[t], label=f"t={t}")
ax.set_xlabel("x (posisi)")
ax.set_ylabel("T (suhu)")
ax.legend()
st.pyplot(fig)

st.markdown("""grafik visualisasi yang dihasilkan menunjukkan adanya perubahan distribusi suhu sepanjang batang 
            dari waktu ke waktu yang diukur dari t=0 dan hanya ada di tengah batang dengan nilai maksimum 100°C 
            (suhu tertinggi dalam rentang celcius). 
            kemudian, dilakukan perhitungan dengan t = 25, t = 50, t = 75, dan t = 99 dan menghasilkan 
            penyebaran suhu yang menyebar dari tengah ke ujung batang.""")

# --- Analisis Awal ---
st.subheader(" Analisis Distribusi Data")

# Heatmap distribusi suhu
st.markdown("Distribusi suhu sepanjang waktu ditampilkan sebagai heatmap:")
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.heatmap(results, cmap="hot", ax=ax2, cbar_kws={'label': 'Suhu'})
ax2.set_xlabel("Posisi (x)")
ax2.set_ylabel("Waktu (t)")
st.pyplot(fig2)

st.markdown("""
            grafik garis yang ditampilkan memperlihatkan perubahan distribusi suhu terhadap posisi batang di beberapa titik tertentu. 
            dimana dapat terlihat suhu tinggi berada di awal di bagian tengah perlahan menyebar ke arah kedua ujung dan 
            menurun secara bertahap. warna yang ditampilkan pada visualisasi menunjukkan tentang evolusi suhu dari waktu ke waktu. 
            warna yang lebih terang menunjukkan suhu yang lebih tinggi dan begitupun sebaliknya""")

# Statistik ringkasan
st.markdown("Statistik ringkasan suhu pada timestep terakhir:")
st.write(pd.Series(results[-1]).describe())

st.markdown("""berdasarkan summary data, suhu maksimum terdapat pada di 99,60°C dengan nilai 
            rata-rata di seluruh batang adalah 26°C yang menandakan penyebaran panas yang 
            menyeluruh dan penurunan suhu di titik titik pusat. standar deviasi yang tinggi 
            menandakan distribusi suhu yang cukup tersebar.""")

# Header
st.header("Runge-Kutta Method for Simulating Predator-Prey Dynamics (Lotka-Volterra)")

# Penjelasan model
st.markdown("""
Model **Lotka-Volterra** menggambarkan interaksi antara dua spesies: mangsa (prey) dan predator.
Persamaan diferensial orde pertama dari model ini:
""")

st.latex(r"""
\begin{cases}
\frac{dx}{dt} = \alpha x - \beta xy \\
\frac{dy}{dt} = \delta xy - \gamma y
\end{cases}
""")

st.markdown("""
Dengan:
- \( x \): populasi mangsa  
- \( y \): populasi predator  
- \( alpha \): tingkat kelahiran mangsa  
- \( beta \): laju pemangsaan  
- \( gamma \): tingkat kematian predator  
- \( delta \): tingkat pertumbuhan predator berdasarkan jumlah mangsa  

Untuk menyelesaikannya secara numerik, kita bisa menggunakan metode **Runge-Kutta Orde 4**:
""")

st.latex(r"""
\begin{aligned}
k_1 &= f(t_n, y_n) \\
k_2 &= f\left(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_1\right) \\
k_3 &= f\left(t_n + \frac{h}{2}, y_n + \frac{h}{2}k_2\right) \\
k_4 &= f(t_n + h, y_n + hk_3) \\
y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}
""")

# Aplikasi
st.subheader(" Aplikasi")
st.markdown("""
Model ini bisa digunakan dalam berbagai aplikasi biologi populasi seperti:
- Mengestimasi interaksi populasi hewan dalam ekosistem
- Simulasi epidemi penyakit (host-pathogen model)
- Strategi pengelolaan satwa liar dan konservasi
""")

# Kode simulasi
st.subheader(" Kode Simulasi (Runge-Kutta 4)")

def lotka_volterra(t, x, y, alpha, beta, gamma, delta):
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return dxdt, dydt

def runge_kutta_lotka_volterra(x0, y0, t0, t_end, h, alpha, beta, gamma, delta):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    t = t0
    x = x0
    y = y0

    while t <= t_end:
        dx1, dy1 = lotka_volterra(t, x, y, alpha, beta, gamma, delta)
        dx2, dy2 = lotka_volterra(t + h/2, x + h*dx1/2, y + h*dy1/2, alpha, beta, gamma, delta)
        dx3, dy3 = lotka_volterra(t + h/2, x + h*dx2/2, y + h*dy2/2, alpha, beta, gamma, delta)
        dx4, dy4 = lotka_volterra(t + h, x + h*dx3, y + h*dy3, alpha, beta, gamma, delta)

        x += h * (dx1 + 2*dx2 + 2*dx3 + dx4) / 6
        y += h * (dy1 + 2*dy2 + 2*dy3 + dy4) / 6
        t += h

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)

    return pd.DataFrame({'Time': t_values, 'Prey': x_values, 'Predator': y_values})

st.markdown("""
```python
            def lotka_volterra(t, x, y, alpha, beta, gamma, delta):
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return dxdt, dydt

def runge_kutta_lotka_volterra(x0, y0, t0, t_end, h, alpha, beta, gamma, delta):
    t_values = [t0]
    x_values = [x0]
    y_values = [y0]

    t = t0
    x = x0
    y = y0

    while t <= t_end:
        dx1, dy1 = lotka_volterra(t, x, y, alpha, beta, gamma, delta)
        dx2, dy2 = lotka_volterra(t + h/2, x + h*dx1/2, y + h*dy1/2, alpha, beta, gamma, delta)
        dx3, dy3 = lotka_volterra(t + h/2, x + h*dx2/2, y + h*dy2/2, alpha, beta, gamma, delta)
        dx4, dy4 = lotka_volterra(t + h, x + h*dx3, y + h*dy3, alpha, beta, gamma, delta)

        x += h * (dx1 + 2*dx2 + 2*dx3 + dx4) / 6
        y += h * (dy1 + 2*dy2 + 2*dy3 + dy4) / 6
        t += h

        t_values.append(t)
        x_values.append(x)
        y_values.append(y)

    return pd.DataFrame({'Time': t_values, 'Prey': x_values, 'Predator': y_values})""")

# Parameter dan hasil simulasi
alpha, beta, gamma, delta = 0.1, 0.02, 0.3, 0.01
x0, y0 = 40, 9
t0, t_end, h = 0, 200, 0.5

df_sim = runge_kutta_lotka_volterra(x0, y0, t0, t_end, h, alpha, beta, gamma, delta)

# Tampilkan data
st.subheader("Data Hasil Simulasi")
st.dataframe(df_sim.head(10))

st.markdown("""
           berikut merupakan tabel hasil simulasi antara mangsa dan predator berdasarkan model Lotka-Volterra dengan menggunakan metode numerik Runge-Kutta. 
            pada tabel terlihat waktu awal dimulai dari t = 0, populasi mangsa = 40, dan predator = 9. 
            pada waktu t = 4.5, jumlah mangsa menurun menjadi sekitar 25,379. 
            penurunan ini menggambarkan bahwa mangsa mengalami tekanan akibat interaksi para predator 
            (mangsa dimakan, namun predator naik karena mendapat cukup asupan untuk hidup dan bereproduksi) """)

# Visualisasi interaktif
fig2 = px.line(df_sim, x="Time", y=["Prey", "Predator"], title="Interactive Population Dynamics")
st.plotly_chart(fig2)

st.markdown("""menunjukkan dinamika populasi mangsa dan predator selama periode waktu tertentu. 
            populasi mangsa mengalami peningkatan tajam hingga mencapai titik puncak lalu mengalami 
            penurunan yang drastis dan biasanya diikuti oleh adanya peningkatan populasi predator. 
            ketika mangsa mulai sedikit, jumlah predator akan menurun dan populasi mangsa bisa kembali tumbuh.""")

# Distribusi
st.subheader("Analisis Awal Distribusi")
fig3, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_sim["Prey"], bins=30, ax=ax[0], color="green", kde=True)
ax[0].set_title("Distribusi Populasi Mangsa (Prey)")
sns.histplot(df_sim["Predator"], bins=30, ax=ax[1], color="red", kde=True)
ax[1].set_title("Distribusi Populasi Predator")
st.pyplot(fig3)

st.markdown("""menunjukkan distribusi populasi dari mangsa dan predator berdasarkan data simulasi. """)


st.header("Runge-Kutta Method for Simulating Finance and Economic Modeling")

# Penjelasan dan persamaan diferensial
st.markdown("""
### Equation: Runge-Kutta 4th Order (RK4)

Metode Runge-Kutta orde-4 adalah salah satu metode numerik untuk menyelesaikan persamaan diferensial biasa (ODE). Berikut adalah formula umum RK4:

$$
\\begin{aligned}
k_1 &= f(t_n, y_n) \\\\
k_2 &= f\\left(t_n + \\frac{h}{2}, y_n + \\frac{h}{2}k_1\\right) \\\\
k_3 &= f\\left(t_n + \\frac{h}{2}, y_n + \\frac{h}{2}k_2\\right) \\\\
k_4 &= f(t_n + h, y_n + hk_3) \\\\
y_{n+1} &= y_n + \\frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\\end{aligned}
$$

Dalam konteks **finance**, metode ini bisa digunakan untuk memodelkan:
- Pertumbuhan investasi (compound interest)
- Model stokastik sederhana (tanpa noise)
- Simulasi pertumbuhan ekonomi berdasarkan model diferensial

### Contoh Aplikasi: Pertumbuhan Investasi dengan Bunga Majemuk
Persamaan: 
$$
\\frac{dP}{dt} = r \\cdot P
$$
dengan:
- \\(P\\): Nilai investasi
- \\(r\\): Tingkat bunga (interest rate)
""")

# Implementasi kode RK4
st.markdown("### Kode Runge-Kutta 4 untuk Simulasi Pertumbuhan Investasi:")

st.code("""
def runge_kutta(f, y0, t0, tn, h):
    n = int((tn - t0) / h)
    t_values = [t0]
    y_values = [y0]
    
    t = t0
    y = y0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        t_values.append(t)
        y_values.append(y)
    return t_values, y_values

# Fungsi pertumbuhan eksponensial
f = lambda t, P: 0.05 * P  # 5% interest rate
t_vals, P_vals = runge_kutta(f, y0=1000, t0=0, tn=10, h=0.1)
""", language="python")

# Jalankan simulasi
def runge_kutta(f, y0, t0, tn, h):
    n = int((tn - t0) / h)
    t_values = [t0]
    y_values = [y0]
    
    t = t0
    y = y0
    for _ in range(n):
        k1 = f(t, y)
        k2 = f(t + h/2, y + h/2 * k1)
        k3 = f(t + h/2, y + h/2 * k2)
        k4 = f(t + h, y + h * k3)
        y = y + (h / 6)*(k1 + 2*k2 + 2*k3 + k4)
        t = t + h
        t_values.append(t)
        y_values.append(y)
    return t_values, y_values

f = lambda t, P: 0.05 * P  # 5% interest rate
t_vals, P_vals = runge_kutta(f, y0=1000, t0=0, tn=10, h=0.1)
df = pd.DataFrame({'Time (years)': t_vals, 'Investment Value': P_vals})

# Tampilkan data simulasi
st.markdown("### Data Hasil Simulasi:")
st.dataframe(df)

st.markdown("""berdasarkan hasil tabel simulasi pertumbuhan investasi, nilai investasi mengalami peningkatan dari waktu ke waktu yang mencerminkan proses eksponensial. sebagai contoh, nilai investasi awal sebesar 1000 mengalami peningkatan yang bertahap, hal ini menunjukkan bunga majemuk bekerja secara kontinu dalam jangka waktu pendek.""")
# Visualisasi
st.markdown("### Visualisasi Pertumbuhan Investasi:")

fig_line = px.line(df, x="Time (years)", y="Investment Value", title="Simulasi Pertumbuhan Investasi (5% Per Tahun)")
st.plotly_chart(fig_line)

# Analisis distribusi
st.markdown("### Analisis Distribusi Data:")
fig_dist, ax = plt.subplots()
sns.histplot(df["Investment Value"], bins=20, kde=True, ax=ax)
ax.set_title("Distribusi Nilai Investasi Setelah Simulasi")
st.pyplot(fig_dist)

st.markdown("""visualisasi menunjukkan pola distribusi yang mendekati normal namun cenderung ke kanan yang disebabkan adanya pertumbuhan eksponensial investasi dengan bunga majemuk. hal ini mencerminkan secara umum investasi tumbuh stabil dengan variasi yang kecil dalam waktu atau laju pertumbuhan yang menghasilkan nilai akhir berbeda. visualisasi ini mendukung asumsi pertumbuhan tetap sebesar 5% per tahun yang hasil akhirnya tersebar karena efek akumulatif bunga majemuk dalam simulasi""")

st.subheader("Evaluation and Discussion")
st.markdown("""
    ### Diskusi
Metode Runge-Kutta memiliki kelebihan sebagai berikut :
- memiliki tingkat keakurasian yang tinggi dibandingkan dengan metode eksplisit sederhana (re. Euler)
- cocok untuk digunakan dalam sistem dinamika nonlinier dan juga fenomena yang kompleks yang memiliki keterlibatan terhadap perubahan waktu
- fleksibel digunakan dalam berbagai bidang, seperti fisika, biologi, hingga perhitungan ekonomi.

namun, selain kelebihan, metode Runge-Kutta juga memiliki kekurangan atau keterbatasan, seperti :

- membutuhkan komputasi yang lebih besar dibanding Euler
- tidak memberikan estimasi error lokal secara langsung
            
### Interpretasi
            
berikut ringkasan interpretasi hasil dari hal yang telah dilakukan sebelumnya :

- **simulasi suhu:** Metode Runge-Kutta dapat menunjukkan bagaimana perubahan suhu secara dinamis. 
- **Simulasi Gerak (Mekanika):** metode ini mampu melacak posisi dan kecepatan objek berdasarkan sistem hukum gerak Newton.
- **Simulasi Aliran Fluida / Konduksi Panas:** metode ini digunakan dalam menstimulasikan distribusi suhu atau aliran partikel dalam waktu
- **Simulasi Predator-Prey (Lotka-Volterra):** mampu memberikan gambaran tentang dinamika ekosistem
- **Simulasi Keuangan dan Ekonomi:** dapat menyelesaikan model ekonomi atau harga saham dengan basis persamaan diferensial

### Kesimpulan

metode ini memiliki keefektifan yang digunakan dalam menyelesaikan model matematika berbasis persamaan diferensial. 
            dalam berbagai konteks membuatnya berguna khususnya dalam bidang sains dan teknik. Runge-Kutta memberikan pemahaman terkait 
            kedinamisan waktu dari sistem kompleks dan membantu memvisualisasikan perilaku sistem dalam berbagai konteks.        
""")
