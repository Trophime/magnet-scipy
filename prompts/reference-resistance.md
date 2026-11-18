Pour les calculs, c'est du copié collé de ce qu'avait fait mon prédécesseur Heink

Calcul de la résistance en fonction du modèle ramené à 0°C et de l'intensité de l'aimant.
// Résistance calculée : R0c = Bi * I^2 + Ai.I + R00
    scoils[k].rcalc = absolu(scoils[k].r00 + scoils[k].ra * scoils[k].icoil + scoils[k].rb * sq(scoils[k].icoil));

Température moyenne de l'eau, foireuse actuellement car avec un retard de dingue sur la valeur en rampe!
/// Température moyenne de l'eau en °C
    // Mean of the temperature between input and output
    scoils[k].tmoy = (tin + fifo_s.tout)/2;


    /// Résistance mesurée ramenée à 0 en mOhms
    // Resistance at 0°C
    // Rm0 = Rm / ( 1 + alpha.T) avec Rm = Um / Im
scoils[k].r0 = absolu(1000.0*scoils[k].ucoil / scoils[k].icoil) / (1 + scoils[k].alpha * scoils[k].tmoy);

les Calculs de dR en %
/// Delta R
    // DeltaR0 = 100.(R0m - R0c) / R0c
    scoils[k].dr0 = 100.0 * (absolu(scoils[k].r0) - absolu(scoils[k].rcalc)) / scoils[k].rcalc;

La température moyenne des hélices pas très correcte car je n'ai pas de bonnes valeurs de alpha (je ne les change quasi jamais alors que le matériaux change donc j'imagine que cette valeur aussi !!!)
/// T équivalent en °C
    // Calculated temperature with U, I and R00
    // Teq = (Rm / R00 - 1)/alpa avec Rm = Um / Im
        scoils[k].tcalc = absolu(absolu((scoils[k].ucoil*1000.0 / scoils[k].icoil)/(scoils[k].r00)) - 1) / scoils[k].alpha;

