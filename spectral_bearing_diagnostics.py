"""
Спектральный анализ затухающих колебаний для диагностики подшипника.
Python 3.x; зависимости: numpy, matplotlib.

Режим работы:
1) Если найдены реальные CWRU-файлы 1797_Normal.npz и 1797_IR_7_DE12.npz,
   скрипт загружает канал DE из них.
2) Если файлов нет, включается воспроизводимая демонстрационная генерация,
   имитирующая структуру задачи: 1797 об/мин, BPFI и затухающие импульсы.

Результаты сохраняются в папку results/:
    fig1_time.png
    fig2_spectrum.png
    fig3_algorithm.png
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------- Параметры эксперимента --------------------
FS = 12_000.0             # Гц, DE12
RPM = 1797.0
FR = RPM / 60.0           # частота вращения, Гц
BPFI_COEFF = 5.415        # коэффициент BPFI для рассматриваемой конфигурации
BPFI = BPFI_COEFF * FR
WINDOW_SECONDS = 2.0
SEED = 42

DATA_DIR = Path("Data")
NORMAL_CANDIDATES = [
    DATA_DIR / "1797_Normal.npz",
    DATA_DIR / "1797 RPM" / "1797_Normal.npz",
    Path("1797_Normal.npz"),
]
FAULT_CANDIDATES = [
    DATA_DIR / "1797_IR_7_DE12.npz",
    DATA_DIR / "1797 RPM" / "1797_IR_7_DE12.npz",
    Path("1797_IR_7_DE12.npz"),
]
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def find_first(paths):
    """Возвращает первый существующий путь из списка или None."""
    return next((p for p in paths if p.exists()), None)


def load_npz_signal(path):
    """
    Загружает первый доступный канал DE из NPZ.
    Если массив многомерный, он разворачивается в одномерный ряд.
    """
    with np.load(path, allow_pickle=False) as data:
        if "DE" in data.files:
            x = data["DE"]
        else:
            # Запасной вариант: взять первый массив NPZ.
            x = data[data.files[0]]
    return np.asarray(x, dtype=float).squeeze().ravel()


def make_demo_signals(fs=FS, seconds=WINDOW_SECONDS, seed=SEED):
    """
    Создает воспроизводимую демонстрационную пару сигналов.
    Нормальный сигнал содержит вращательную составляющую и шум.
    Дефектный сигнал дополнительно содержит периодические
    затухающие импульсы на резонансной частоте.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs

    normal = (
        0.50 * np.sin(2 * np.pi * FR * t)
        + 0.18 * np.sin(2 * np.pi * 2 * FR * t + 0.3)
        + 0.08 * np.sin(2 * np.pi * 3 * FR * t)
        + 0.05 * rng.standard_normal(n)
    )

    fault = normal.copy()
    period = max(1, int(round(fs / BPFI)))

    # Периодические затухающие колебания от локального дефекта.
    for start in range(0, n, period):
        length = min(int(0.020 * fs), n - start)
        tau = np.arange(length) / fs
        ringdown = (
            0.95
            * np.exp(-tau / 0.004)
            * np.sin(2 * np.pi * 2200 * tau)
        )
        fault[start:start + length] += ringdown

    fault += 0.04 * rng.standard_normal(n)
    return t, normal, fault


def prepare(x, n=None):
    """Удаление среднего и ограничение длины анализируемого окна."""
    x = np.asarray(x, dtype=float).ravel()
    if n is not None:
        x = x[:n]
    x = x - np.mean(x)
    return x


def amplitude_spectrum(x, fs):
    """Односторонний амплитудный спектр с окном Хэннинга."""
    w = np.hanning(len(x))
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(len(x), 1.0 / fs)
    amplitude = 2.0 * np.abs(X) / np.sum(w)
    return f, amplitude


def diagnostic_ratio(f, a, target=BPFI, search_half_width=8.0):
    """
    Ищет локальный максимум вокруг BPFI и сравнивает его
    с медианой амплитуд в соседней фоновой полосе.
    """
    target_band = (f >= target - search_half_width) & (
        f <= target + search_half_width
    )
    background_band = (f >= target - 60) & (f <= target - 20)

    peak_index = np.argmax(a[target_band])
    peak_freq = f[target_band][peak_index]
    peak_amp = a[target_band][peak_index]

    background = np.median(a[background_band])
    ratio = peak_amp / max(background, np.finfo(float).eps)

    return peak_freq, peak_amp, background, ratio


def main():
    normal_path = find_first(NORMAL_CANDIDATES)
    fault_path = find_first(FAULT_CANDIDATES)

    if normal_path and fault_path:
        # Реальные данные CWRU.
        normal = load_npz_signal(normal_path)
        fault = load_npz_signal(fault_path)
        n = min(len(normal), len(fault), int(FS * WINDOW_SECONDS))
        normal = prepare(normal, n)
        fault = prepare(fault, n)
        t = np.arange(n) / FS
        source = "CWRU NPZ"
    else:
        # Демонстрационный режим, если NPZ-файлы не размещены рядом со скриптом.
        t, normal, fault = make_demo_signals()
        source = "демонстрационная модель"

    fn, an = amplitude_spectrum(normal, FS)
    ff, af = amplitude_spectrum(fault, FS)

    peak_f, peak_a, background, ratio = diagnostic_ratio(ff, af)

    # ---------- Рисунок 1: временные реализации ----------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mask = t <= min(0.25, t[-1])
    ax.plot(t[mask], normal[mask], label="Исправное состояние", linewidth=1)
    ax.plot(t[mask], fault[mask], label="Дефект внутреннего кольца",
            linewidth=1, alpha=0.85)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Амплитуда, усл. ед.")
    ax.set_title("Временные реализации вибрационных сигналов")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_time.png", dpi=300)
    plt.close(fig)

    # ---------- Рисунок 2: амплитудные спектры ----------
    fig, ax = plt.subplots(figsize=(8, 4.8))
    mask_n = fn <= 1000
    mask_f = ff <= 1000
    ax.plot(fn[mask_n], an[mask_n], label="Исправное состояние",
            linewidth=1)
    ax.plot(ff[mask_f], af[mask_f], label="Дефект внутреннего кольца",
            linewidth=1)
    ax.axvline(BPFI, linestyle="--", linewidth=1,
               label=f"BPFI = {BPFI:.1f} Гц")
    ax.set_xlabel("Частота, Гц")
    ax.set_ylabel("Амплитуда, усл. ед.")
    ax.set_title("Амплитудные спектры")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_spectrum.png", dpi=300)
    plt.close(fig)

    # ---------- Рисунок 3: блок-схема ----------
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.axis("off")
    steps = [
        ("Вход: x(t)", 0.92),
        ("Удаление среднего\nи окно Хэннинга", 0.76),
        ("Оконное FFT", 0.60),
        ("Поиск пика около BPFI", 0.44),
        ("R = A(BPFI) / A_фон", 0.28),
        ("R > порога?", 0.12),
    ]
    for text, y in steps:
        ax.text(
            0.5, y, text, ha="center", va="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="white", edgecolor="black")
        )

    for y1, y2 in zip([0.88, 0.72, 0.56, 0.40, 0.24],
                       [0.80, 0.64, 0.48, 0.32, 0.20]):
        ax.annotate("", xy=(0.5, y2), xytext=(0.5, y1),
                    arrowprops=dict(arrowstyle="->", lw=1))

    ax.text(0.70, 0.12, "Да → ДЕФЕКТ", ha="left", va="center", fontsize=11)
    ax.text(0.30, 0.12, "Нет → НОРМА", ha="right", va="center", fontsize=11)

    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_algorithm.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Источник данных: {source}")
    print(f"f_r = {FR:.3f} Гц")
    print(f"BPFI = {BPFI:.3f} Гц")
    print(f"Найденный пик = {peak_f:.3f} Гц")
    print(f"A_BPFI = {peak_a:.6f}")
    print(f"A_фон = {background:.6f}")
    print(f"R = {ratio:.3f}")
    print(f"Графики сохранены в: {RESULTS.resolve()}")


if __name__ == "__main__":
    main()
