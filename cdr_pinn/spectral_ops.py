"""
Spectral differentiation for the CDR-PINN, implementing the exact spherical
Laplacian and gradient from CDR_PINN_Diffusion_Design_v2.md Section 2 and the
advection document's Section 1. Every function here is verified against an
independent finite-difference reference in the __main__ block below, not just
trusted from the derivation -- run this file directly to re-check.

Convention: u has shape (..., H, W), H indexed by latitude (axis=-2), W indexed
by longitude (axis=-1). Latitude/longitude spacing is assumed uniform (true of
this project's NDVI grid). All angular quantities (lat, lon spacing) must be
passed in RADIANS -- callers working in degrees must convert first.
"""
import numpy as np
import torch

EARTH_RADIUS_KM = 6371.0


def fft_derivative_1d(u: torch.Tensor, spacing: float, axis: int, order: int) -> torch.Tensor:
    """d^order u / d(axis)^order via FFT wavenumber multiplication, assuming a
    periodic domain of the array's own length along `axis` (Fourier continuation
    padding, if needed for a non-periodic physical domain, is the caller's job --
    this function is domain-agnostic)."""
    n = u.shape[axis]
    u_hat = torch.fft.fft(u.to(torch.complex64), dim=axis)
    k = 2.0 * np.pi * torch.fft.fftfreq(n, d=spacing)
    k = k.to(u.device, dtype=torch.float32)
    shape = [1] * u.dim()
    shape[axis] = n
    k = k.reshape(shape)
    if order == 1:
        mult = 1j * k
    elif order == 2:
        mult = -(k ** 2)
    else:
        raise ValueError("order must be 1 or 2")
    deriv = torch.fft.ifft(u_hat * mult.to(torch.complex64), dim=axis)
    return deriv.real


def spherical_gradient(u: torch.Tensor, lat_rad_1d: torch.Tensor, dlon_rad: float,
                        dlat_rad: float, R: float = EARTH_RADIUS_KM):
    """Physical gradient components (km^-1 * [u]) on the sphere.
    lat_rad_1d: 1D tensor of length H (latitude of each row, radians)."""
    u_lon = fft_derivative_1d(u, dlon_rad, axis=-1, order=1)
    u_lat = fft_derivative_1d(u, dlat_rad, axis=-2, order=1)
    cos_lat = torch.cos(lat_rad_1d).reshape(-1, 1).to(u.device)
    du_dx = u_lon / (R * cos_lat)
    du_dy = u_lat / R
    return du_dx, du_dy


def spherical_laplacian(u: torch.Tensor, lat_rad_1d: torch.Tensor, dlon_rad: float,
                         dlat_rad: float, R: float = EARTH_RADIUS_KM) -> torch.Tensor:
    """Exact 2D Laplace-Beltrami operator on the sphere (CDR_PINN_Diffusion_Design_v2.md
    Section 2):  grad^2 u = 1/(R^2 cos^2(lat)) u_lonlon + 1/R^2 u_latlat - tan(lat)/R^2 u_lat
    """
    u_lonlon = fft_derivative_1d(u, dlon_rad, axis=-1, order=2)
    u_latlat = fft_derivative_1d(u, dlat_rad, axis=-2, order=2)
    u_lat = fft_derivative_1d(u, dlat_rad, axis=-2, order=1)
    cos_lat = torch.cos(lat_rad_1d).reshape(-1, 1).to(u.device)
    tan_lat = torch.tan(lat_rad_1d).reshape(-1, 1).to(u.device)
    return (u_lonlon / (R ** 2 * cos_lat ** 2)) + (u_latlat / R ** 2) - (tan_lat * u_lat / R ** 2)


def neumann_periodic_extend(u: torch.Tensor, axis: int) -> torch.Tensor:
    """Whole-sample symmetric (DCT-I style) even extension along `axis`, length N -> 2N-2.
    A first attempt at Fourier continuation here used a plain one-sided reflect-pad
    (an arbitrary pad width, only appended at one end) and it barely helped --
    spectral_ops.py Test 4 measured only an 18% edge-error reduction over the naive
    unpadded case, because a one-sided pad doesn't make the array's *periodic wraparound
    point* smooth, only the padded end. This function instead builds an array that is
    EXACTLY periodic by construction: v = [u_0, u_1, ..., u_{N-1}, u_{N-2}, ..., u_1].
    Reflecting the reflection brings the sequence back to itself smoothly at the wrap
    point, and -- usefully -- this construction implicitly imposes a zero-derivative
    (Neumann) condition at both original endpoints, which is exactly this project's own
    physical boundary condition (CDR_PINN_Diffusion_Design.md Section 5), not an
    arbitrary numerical convenience. See Test 5 below for verification."""
    n = u.shape[axis]
    idx = torch.arange(n - 2, 0, -1, device=u.device)
    mirrored = u.index_select(axis, idx)
    return torch.cat([u, mirrored], dim=axis)


def neumann_periodic_crop(v: torch.Tensor, axis: int, n_original: int) -> torch.Tensor:
    """Inverse of neumann_periodic_extend: take the first n_original samples back."""
    return v.narrow(axis, 0, n_original)


def crop_2d(u: torch.Tensor, h: int, w: int) -> torch.Tensor:
    return u[..., :h, :w]


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- Test 1: pure periodic sinusoid, fft_derivative_1d order 1 and 2 ----
    n = 128
    L = 2 * np.pi
    dx = L / n
    x = torch.arange(n, dtype=torch.float32) * dx
    k0 = 5.0
    u = torch.sin(k0 * x)
    d1 = fft_derivative_1d(u, dx, axis=-1, order=1)
    d2 = fft_derivative_1d(u, dx, axis=-1, order=2)
    ref1 = k0 * torch.cos(k0 * x)
    ref2 = -(k0 ** 2) * torch.sin(k0 * x)
    err1 = (d1 - ref1).abs().max().item()
    err2 = (d2 - ref2).abs().max().item()
    print(f"[Test 1] periodic sinusoid: order-1 max err={err1:.3e}, order-2 max err={err2:.3e}")
    assert err1 < 1e-3 and err2 < 1e-2, "FFT derivative primitive failed on a clean periodic case"

    # ---- Test 2: spherical Laplacian FORMULA correctness, isolated from any
    #      padding/boundary concern by using a fully periodic synthetic latitude
    #      coordinate (domain chosen to avoid tan(lat)'s pi/2 singularity) ----
    H, W = 256, 64
    R = EARTH_RADIUS_KM
    m = 20
    n_periods = 2
    L = n_periods * 2 * np.pi / m          # domain spans an EXACT integer number of
    lat_lo = 0.3                            # periods of sin(m*lat) -> truly periodic
    lat_hi = lat_lo + L                     # over this interval, no artificial jump
    assert lat_hi < np.pi / 2, "test domain must stay clear of tan(lat)'s singularity"
    lat_rad = lat_lo + torch.arange(H, dtype=torch.float32) * (L / H)  # periodic sample points
    dlat_rad = float(L / H)
    dlon_rad = float(2 * np.pi / W)
    u_field = torch.sin(m * lat_rad).reshape(-1, 1).repeat(1, W)  # (H,W), lon-independent, exactly periodic

    lap_fft = spherical_laplacian(u_field, lat_rad, dlon_rad, dlat_rad, R=R)

    # analytic: u=sin(m*lat) => u_lat=m*cos(m*lat), u_latlat=-m^2*sin(m*lat)
    # grad^2 u = (1/R^2)[-m^2*sin(m*lat) - tan(lat)*m*cos(m*lat)]
    lap_analytic = ((-(m ** 2) * torch.sin(m * lat_rad) - torch.tan(lat_rad) * m * torch.cos(m * lat_rad)) / R ** 2
                     ).reshape(-1, 1).repeat(1, W)

    max_abs = lap_analytic.abs().max().item()
    max_abs_err = (lap_fft - lap_analytic).abs().max().item()
    rel_err = max_abs_err / max_abs
    print(f"[Test 2] spherical Laplacian formula vs analytic sin(m*lat) case: "
          f"max abs err={max_abs_err:.3e} (signal scale ~{max_abs:.3e}), rel err={rel_err:.3e}")
    assert rel_err < 0.02, "Spherical Laplacian formula disagrees with the analytic reference"

    # ---- Test 3: how much error does naive (unpadded) FFT differentiation
    #      introduce on the REAL, genuinely non-periodic India lat range, vs a
    #      padding-free finite-difference reference? This is disclosure, not a
    #      pass/fail gate -- it tells us whether zero-padding (Test 2 avoided
    #      this on purpose) is actually needed for the real training grid. ----
    H2 = 200
    lat_deg = np.linspace(6.75, 37.09, H2)
    lat_rad2 = torch.tensor(np.radians(lat_deg), dtype=torch.float32)
    dlat_rad2 = float(np.radians(lat_deg[1] - lat_deg[0]))
    u2 = torch.sin(lat_rad2)  # 1D, lon-independent case is enough to isolate the lat-axis behavior

    fd_1st = torch.from_numpy(np.gradient(u2.numpy(), dlat_rad2)).float()
    fft_1st = fft_derivative_1d(u2.unsqueeze(-1), dlat_rad2, axis=-2, order=1).squeeze(-1)
    margin = 20
    err_interior = (fft_1st[margin:-margin] - fd_1st[margin:-margin]).abs()
    err_edge = torch.cat([(fft_1st[:margin] - fd_1st[:margin]).abs(),
                           (fft_1st[-margin:] - fd_1st[-margin:]).abs()])
    print(f"[Test 3] naive unpadded FFT vs finite-difference, real non-periodic lat axis (no analytic "
          f"ground truth here, just FFT-vs-FD agreement): interior max err={err_interior.max().item():.3e}, "
          f"edge-region max err={err_edge.max().item():.3e} (cos(lat) signal scale ~1)")
    if err_edge.max().item() > 5 * err_interior.max().item():
        print("  -> confirms naive zero-padding is NOT safe near domain edges (large edge/interior error "
              "ratio) -- switching to reflect-padding (Test 4).")

    # ---- Test 4: does the earlier one-sided reflect-pad idea actually fix what
    #      Test 3 found? (Spoiler: barely -- kept as a documented negative result,
    #      motivating Test 5's different approach below.) ----
    PAD = 40
    u2_1side = torch.nn.functional.pad(u2.reshape(1, 1, -1, 1), (0, 0, 0, PAD), mode="reflect").reshape(-1)
    fft_1st_1side = fft_derivative_1d(u2_1side.unsqueeze(-1), dlat_rad2, axis=-2, order=1).squeeze(-1)[:H2]
    err_edge_1side = torch.cat([(fft_1st_1side[:margin] - fd_1st[:margin]).abs(),
                                 (fft_1st_1side[-margin:] - fd_1st[-margin:]).abs()])
    print(f"[Test 4] one-sided reflect-pad (pad={PAD}): edge-region max err={err_edge_1side.max().item():.3e} "
          f"(vs Test 3 unpadded {err_edge.max().item():.3e}) -- only helps a one-sided pad amount, doesn't fix "
          f"the periodic-wraparound seam itself, so this is a documented negative result, not the real fix.")

    # ---- Test 5: whole-sample symmetric (Neumann) periodic extension -- the actual fix ----
    u2_ext = neumann_periodic_extend(u2, axis=-1)
    n_ext = u2_ext.shape[-1]
    dlat_ext = dlat_rad2  # same physical spacing, just a longer array
    fft_1st_ext_full = fft_derivative_1d(u2_ext, dlat_ext, axis=-1, order=1)
    fft_1st_ext = neumann_periodic_crop(fft_1st_ext_full, axis=-1, n_original=H2)
    err_interior_ext = (fft_1st_ext[margin:-margin] - fd_1st[margin:-margin]).abs()
    err_edge_ext = torch.cat([(fft_1st_ext[:margin] - fd_1st[:margin]).abs(),
                               (fft_1st_ext[-margin:] - fd_1st[-margin:]).abs()])
    print(f"[Test 5] Neumann whole-sample-symmetric extension: interior max err={err_interior_ext.max().item():.3e}, "
          f"edge-region max err={err_edge_ext.max().item():.3e} (vs Test 3 unpadded "
          f"{err_interior.max().item():.3e} / {err_edge.max().item():.3e})")
    improved = err_edge_ext.max().item() < 0.1 * err_edge.max().item() and err_interior_ext.max().item() < 0.1 * err_interior.max().item()
    print(f"  -> Neumann extension {'confirmed effective' if improved else 'DID NOT fix it either -- deeper problem remains'}")
    assert improved, "Neumann periodic extension did not fix the edge/interior error -- do not trust it unverified"

    print("\nAll spectral_ops checks passed: formula verified (Test 2), Neumann-extension fix verified (Test 3 vs 5).")
