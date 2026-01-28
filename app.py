from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from flask import Flask, render_template, request
from sympy import Matrix, Rational, latex, nsimplify

app = Flask(__name__)


def _to_rational(s: str) -> Rational:
    """
    Parse user input into an exact Rational where possible.
    Accepts integers, decimals, and fractions like 3/5.
    """
    s = (s or "").strip()
    if s == "":
        return Rational(0)
    # nsimplify handles "0.1" -> 1/10, "3/5" -> 3/5, etc.
    return Rational(nsimplify(s))


def _read_matrix(prefix: str, rows: int, cols: int) -> Matrix:
    data: list[list[Rational]] = []
    for r in range(1, rows + 1):
        row: list[Rational] = []
        for c in range(1, cols + 1):
            row.append(_to_rational(request.form.get(f"{prefix}{r}{c}", "")))
        data.append(row)
    return Matrix(data)


def _matrix_latex(M: Matrix) -> str:
    return r"\begin{bmatrix}" + r"\\".join(
        [" & ".join([latex(v) for v in M.row(i)]) for i in range(M.rows)]
    ) + r"\end{bmatrix}"


@dataclass
class Result:
    title: str
    steps_latex: list[str]


def _mul_steps(C: Matrix, A: Matrix) -> list[str]:
    steps: list[str] = []
    steps.append(r"Hitung \(CA\) dengan perkalian baris-kolom.")
    steps.append(rf"\(C={_matrix_latex(C)}\), \(A={_matrix_latex(A)}\)")
    out = C * A
    for i in range(out.rows):
        for j in range(out.cols):
            terms = [rf"({latex(C[i,k])})({latex(A[k,j])})" for k in range(C.cols)]
            expanded = " + ".join(terms) if terms else "0"
            steps.append(
                rf"\((CA)_{{{i+1}{j+1}}} = {expanded} = {latex(out[i,j])}\)"
            )
    steps.append(rf"\(\Rightarrow CA = {_matrix_latex(out)}\)")
    return steps


def _transpose_plus_steps(A: Matrix, B: Matrix) -> list[str]:
    steps: list[str] = []
    steps.append(r"Hitung \(A^T + B\).")
    steps.append(rf"\(A={_matrix_latex(A)}\), \(B={_matrix_latex(B)}\)")
    At = A.T
    steps.append(rf"\(A^T = {_matrix_latex(At)}\)")
    out = At + B
    for i in range(out.rows):
        for j in range(out.cols):
            steps.append(
                rf"\((A^T+B)_{{{i+1}{j+1}}} = {latex(At[i,j])} + {latex(B[i,j])} = {latex(out[i,j])}\)"
            )
    steps.append(rf"\(\Rightarrow A^T + B = {_matrix_latex(out)}\)")
    return steps


def _cbt_steps(C: Matrix, B: Matrix) -> list[str]:
    steps: list[str] = []
    steps.append(r"Hitung \((CB)^T\).")
    steps.append(rf"\(C={_matrix_latex(C)}\), \(B={_matrix_latex(B)}\)")
    CB = C * B
    steps.append(rf"\(CB = {_matrix_latex(CB)}\)")
    out = CB.T
    steps.append(rf"\((CB)^T = {_matrix_latex(out)}\)")
    return steps


def _gauss_jordan_inverse_steps(A: Matrix) -> tuple[Matrix, list[str]]:
    if A.rows != A.cols:
        raise ValueError("Matriks harus persegi (n×n) untuk punya invers.")

    n = A.rows
    M = A.applyfunc(lambda v: Rational(v))  # ensure Rational
    aug = M.row_join(Matrix.eye(n))

    steps: list[str] = []
    steps.append(r"Gunakan metode Operasi Baris Elementer (Gauss-Jordan).")
    steps.append(rf"Mulai dari \([A \mid I]\): \(\left[{_matrix_latex(aug)}\right)\)")  # fallback if brackets odd

    # We'll render augmented matrix in two blocks for nicer latex
    def aug_latex(mat: Matrix) -> str:
        left = mat[:, :n]
        right = mat[:, n:]
        colspec = ("c" * n) + "|" + ("c" * n)
        rows = []
        for i in range(n):
            rows.append(
                " & ".join([latex(v) for v in left.row(i)])
                + " & "
                + " & ".join([latex(v) for v in right.row(i)])
            )
        return r"\left[\begin{array}{" + colspec + r"}" + r"\\".join(rows) + r"\end{array}\right]"

    steps[-1] = rf"Mulai dari \([A \mid I]\): \( {aug_latex(aug)} \)"

    # Gauss-Jordan elimination
    for col in range(n):
        # find pivot
        pivot_row = None
        for r in range(col, n):
            if aug[r, col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError("Determinannya 0, jadi matriks tidak punya invers.")

        if pivot_row != col:
            aug.row_swap(pivot_row, col)
            steps.append(rf"Tukar \(R_{pivot_row+1}\) ↔ \(R_{col+1}\): \( {aug_latex(aug)} \)")

        pivot = aug[col, col]
        if pivot != 1:
            aug.row_op(col, lambda v, j: v / pivot)
            steps.append(
                rf"Buat pivot 1: \(R_{col+1} \leftarrow \frac{{1}}{{{latex(pivot)}}}R_{col+1}\): \( {aug_latex(aug)} \)"
            )

        for r in range(n):
            if r == col:
                continue
            factor = aug[r, col]
            if factor == 0:
                continue
            aug.row_op(r, lambda v, j: v - factor * aug[col, j])
            steps.append(
                rf"Eliminasi kolom {col+1}: \(R_{r+1} \leftarrow R_{r+1} - ({latex(factor)})R_{col+1}\): \( {aug_latex(aug)} \)"
            )

    inv = aug[:, n:]
    steps.append(rf"Karena kiri menjadi \(I\), maka inversnya adalah \(A^{{-1}} = {_matrix_latex(inv)}\).")
    return inv, steps


def _cramer_steps(coeff: Matrix, const: Matrix, var_names: list[str]) -> tuple[Matrix, list[str]]:
    if coeff.rows != coeff.cols:
        raise ValueError("Aturan Cramer butuh matriks koefisien persegi (n×n).")
    if const.cols != 1 or const.rows != coeff.rows:
        raise ValueError("Vektor konstanta harus berukuran n×1.")

    n = coeff.rows
    detA = coeff.det()
    steps: list[str] = []
    steps.append(r"Gunakan Aturan Cramer.")
    steps.append(rf"Matriks koefisien \(A={_matrix_latex(coeff)}\) dan \(b={_matrix_latex(const)}\).")
    steps.append(rf"\(\det(A) = {latex(detA)}\).")
    if detA == 0:
        steps.append(r"Karena \(\det(A)=0\), Aturan Cramer tidak bisa dipakai (solusi tidak tunggal atau tidak ada).")
        return Matrix.zeros(n, 1), steps

    sol = Matrix.zeros(n, 1)
    for i in range(n):
        Ai = coeff.copy()
        Ai[:, i] = const
        detAi = Ai.det()
        sol[i, 0] = detAi / detA
        steps.append(rf"Ganti kolom ke-{i+1} dengan \(b\): \(A_{var_names[i]} = {_matrix_latex(Ai)}\).")
        steps.append(rf"\(\det(A_{var_names[i]}) = {latex(detAi)}\).")
        steps.append(rf"\(\displaystyle {var_names[i]} = \frac{{\det(A_{var_names[i]})}}{{\det(A)}} = \frac{{{latex(detAi)}}}{{{latex(detA)}}} = {latex(sol[i,0])}\).")

    steps.append(rf"\(\Rightarrow \text{{Solusi}} = {_matrix_latex(sol)}\) (urut: {', '.join(var_names)}).")
    return sol, steps


DEFAULTS: dict[str, Any] = {
    "A": [[2, -3], [4, 2]],
    "B": [[-2, 4], [1, 7]],
    "C": [[4, 7], [5, -2], [-1, 1]],
    # Invers presets (yang jelas dari gambar: Kelompok 2-4 adalah 3x3)
    "inv_group": "2",
    "invA2": [[-2, 6, -1], [0, 3, 2], [-3, 2, -2]],
    "invA3": [[1, 0, 1], [1, -3, 0], [-1, 4, 2]],
    "invA4": [[-1, 4, 2], [2, 7, -4], [3, -9, 11]],
    # Cramer presets
    "cramer_group": "1",
}


def _matrix_from_defaults(key: str) -> Matrix:
    return Matrix(DEFAULTS[key]).applyfunc(Rational)


@app.route("/", methods=["GET", "POST"])
def index():
    result: Result | None = None
    active_tab = request.form.get("tab", "matrix") if request.method == "POST" else "matrix"

    # Base matrices (soal no.1)
    A = _matrix_from_defaults("A")
    B = _matrix_from_defaults("B")
    C = _matrix_from_defaults("C")

    # Inverse preset selected
    inv_group = request.form.get("inv_group", DEFAULTS["inv_group"])
    invA = _matrix_from_defaults(f"invA{inv_group}") if inv_group in {"2", "3", "4"} else _matrix_from_defaults("invA2")

    # Cramer preset selected
    cramer_group = request.form.get("cramer_group", DEFAULTS["cramer_group"])

    if request.method == "POST":
        if active_tab == "matrix":
            try:
                op = request.form.get("matrix_op", "CA")
                # allow editing values
                A = _read_matrix("a", 2, 2)
                B = _read_matrix("b", 2, 2)
                C = _read_matrix("c", 3, 2)

                if op == "CA":
                    result = Result("CA", _mul_steps(C, A))
                elif op == "ATB":
                    result = Result(r"A^T + B", _transpose_plus_steps(A, B))
                elif op == "CBT":
                    result = Result(r"(CB)^T", _cbt_steps(C, B))
                else:
                    result = Result("Operasi tidak dikenal", [r"Operasi tidak dikenali."])
            except Exception as e:
                result = Result("Operasi Matriks", [rf"Error: {str(e)}"])

        elif active_tab == "inverse":
            # either preset or custom size 3
            inv_group = request.form.get("inv_group", DEFAULTS["inv_group"])
            if request.form.get("inv_source", "preset") == "preset":
                invA = _matrix_from_defaults(f"invA{inv_group}") if inv_group in {"2", "3", "4"} else _matrix_from_defaults("invA2")
            else:
                # custom 3x3
                invA = _read_matrix("inv", 3, 3)
            try:
                inv, steps = _gauss_jordan_inverse_steps(invA)
                result = Result("Invers Matriks (OBE)", steps)
            except Exception as e:
                result = Result("Invers Matriks (OBE)", [rf"Error: {str(e)}"])

        elif active_tab == "cramer":
            cramer_group = request.form.get("cramer_group", DEFAULTS["cramer_group"])
            try:
                if cramer_group == "1":
                    # 3x + 4y - 5z = 12; 2x + 5y + z = 17; 6x - 2y + 3z = 17
                    coeff = Matrix([[3, 4, -5], [2, 5, 1], [6, -2, 3]]).applyfunc(Rational)
                    const = Matrix([[12], [17], [17]]).applyfunc(Rational)
                    names = ["x", "y", "z"]
                elif cramer_group == "2":
                    # x + y + z = 45; x - y = -4; -x + z = 17
                    coeff = Matrix([[1, 1, 1], [1, -1, 0], [-1, 0, 1]]).applyfunc(Rational)
                    const = Matrix([[45], [-4], [17]]).applyfunc(Rational)
                    names = ["x", "y", "z"]
                elif cramer_group == "3":
                    # a + b - c = 1; 3a - b + c = 0; a - 3b + 3c = -2
                    coeff = Matrix([[1, 1, -1], [3, -1, 1], [1, -3, 3]]).applyfunc(Rational)
                    const = Matrix([[1], [0], [-2]]).applyfunc(Rational)
                    names = ["a", "b", "c"]
                elif cramer_group == "4":
                    # x + 2y = 4; 2x + z = 5; y - 3z = -6
                    coeff = Matrix([[1, 2, 0], [2, 0, 1], [0, 1, -3]]).applyfunc(Rational)
                    const = Matrix([[4], [5], [-6]]).applyfunc(Rational)
                    names = ["x", "y", "z"]
                else:
                    raise ValueError("Kelompok tidak dikenal.")

                _, steps = _cramer_steps(coeff, const, names)
                result = Result("SPL (Aturan Cramer)", steps)
            except Exception as e:
                result = Result("SPL (Aturan Cramer)", [rf"Error: {str(e)}"])

    return render_template(
        "index.html",
        active_tab=active_tab,
        result=result,
        A=A,
        B=B,
        C=C,
        inv_group=inv_group,
        invA=invA,
        cramer_group=cramer_group,
    )


