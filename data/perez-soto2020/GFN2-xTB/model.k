$scheme
 // Reactants                                              Reactants' adducts
 Benzaldehyde(dcm) +   NButylamine(dcm)                <=> A_N(dcm)                            // black
 Benzaldehyde(dcm) + 2 NButylamine(dcm)                <=> A_N_N(dcm)                          // blue
 Benzaldehyde(dcm) +   NButylamine(dcm) + Water(dcm)   <=> A_N_W(dcm)                          // red
 Benzaldehyde(dcm) + 2 NButylamine(dcm) + Water(dcm)   <=> A_N_N_W(dcm)                        // purple
 Benzaldehyde(dcm) +   NButylamine(dcm) + 2 Water(dcm) <=> A_N_W_W(dcm)                        // green

 // Transition states, intermediates and adducts
 A_N(dcm)     -> TS1_#(dcm)   -> Hemiaminal(dcm) -> TS2_#(dcm)   -> I_W(dcm)                   // black
 A_N_N(dcm)   -> TS1N_#(dcm)  -> Int_N(dcm)      -> TS2N_#(dcm)  -> I_N_W(dcm)                 // blue
 A_N_W(dcm)   -> TS1W_#(dcm)  -> Int_W(dcm)      -> TS2W_#(dcm)  -> I_W_W(dcm)                 // red
 A_N_N_W(dcm) -> TS1NW_#(dcm) -> Int_N_W(dcm)    -> TS2NW_#(dcm) -> I_N_W_W(dcm)               // purple
 A_N_W_W(dcm) -> TS1WW_#(dcm) -> Int_W_W(dcm)    -> TS2WW_#(dcm) -> I_W_W_W(dcm)               // green

 // Products' adducts                       Products
 I_W(dcm)                                    <=> Imine(dcm) +   Water(dcm)                     // black
 I_N_W(dcm)                                  <=> Imine(dcm) +   Water(dcm) + NButylamine(dcm)  // blue
 I_W_W(dcm)                                  <=> Imine(dcm) + 2 Water(dcm)                     // red
 I_N_W_W(dcm)                                <=> Imine(dcm) + 2 Water(dcm) + NButylamine(dcm)  // purple
 I_W_W_W(dcm)                                <=> Imine(dcm) + 3 Water(dcm)                     // green
$end

$compounds
 Benzaldehyde(dcm): Benzaldehyde.out
 NButylamine(dcm): NButylamine.out
 Water(dcm): Water.out
 A_N(dcm): A_N.out
 A_N_N(dcm): A_N_N.out
 A_N_W(dcm): A_N_W.out
 A_N_N_W(dcm): A_N_N_W.out
 A_N_W_W(dcm): A_N_W_W.out
 Hemiaminal(dcm): Hemiaminal.out
 Int_N(dcm): Int_N.out
 Int_W(dcm): Int_W.out
 Int_N_W(dcm): Int_N_W.out
 Int_W_W(dcm): Int_W_W.out
 I_W(dcm): I_W.out
 I_N(dcm): I_N.out
 I_N_W(dcm): I_N_W.out
 I_W_W(dcm): I_W_W.out
 I_N_W_W(dcm): I_N_W_W.out
 I_W_W_W(dcm): I_W_W_W.out
 Imine(dcm): Imine.out
 TS1_#(dcm): TS1.out
 TS1N_#(dcm): TS1N.out
 TS1W_#(dcm): TS1W.out
 TS1NW_#(dcm): TS1NW.out
 TS1WW_#(dcm): TS1WW.out
 TS2_#(dcm): TS2.out
 TS2N_#(dcm): TS2N.out
 TS2W_#(dcm): TS2W.out
 TS2NW_#(dcm): TS2NW.out
 TS2WW_#(dcm): TS2WW.out
$end
