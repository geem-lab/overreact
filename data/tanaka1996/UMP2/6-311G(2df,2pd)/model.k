// Radical hydrogen abstraction from methane by chlorine atom
// ==========================================================
//
// The results should be compared with Tanaka (1996) [1].
//
// [1]: https://doi.org/10.1007/BF00058703
//
$scheme
 CH4 + Cl· -> H3CHCl‡ -> CH3· + HCl
$end

$compounds
 CH4: methane.out
 CH3·: CH3·.out
 HCl: HCl.out
 Cl·: Cl·.out
 H3CHCl‡:
  logfile=H3CHCl‡.out
  symmetry=4
$end
