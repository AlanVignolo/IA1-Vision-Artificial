def Hu_Moments(moments):
    # Extraer los momentos centrales desde el diccionario de momentos
    m00 = moments['m00']
    m10 = moments['m10']
    m01 = moments['m01']
    m20 = moments['m20']
    m02 = moments['m02']
    m11 = moments['m11']
    m30 = moments['m30']
    m21 = moments['m21']
    m12 = moments['m12']
    m03 = moments['m03']

    # Calcular el centroide
    xc = m10 / m00
    yc = m01 / m00

    # Calcular los momentos centrales normalizados
    n20 = m20 / m00**2
    n02 = m02 / m00**2
    n11 = m11 / m00**2
    n30 = m30 / m00**2.5
    n12 = m12 / m00**2.5
    n21 = m21 / m00**2.5
    n03 = m03 / m00**2.5

    # Calcular los momentos de Hu
    phi1 = n20 + n02
    phi2 = (n20 - n02)**2 + 4*n11**2
    phi3 = (n30 - 3*n12)**2 + (3*n21 - n03)**2
    phi4 = (n30 + n12)**2 + (n21 + n03)**2
    phi5 = (n30 - 3*n12) * (n30 + n12) * ((n30 + n12)**2 - 3*(n21 + n03)**2) + (3*n21 - n03) * (n21 + n03) * (3*(n30 + n12)**2 - (n21 + n03)**2)
    phi6 = (n20 - n02) * ((n30 + n12)**2 - (n21 + n03)**2) + 4*n11 * (n30 + n12) * (n21 + n03)
    phi7 = (3*n21 - n03) * (n30 + n12) * ((n30 + n12)**2 - 3*(n21 + n03)**2) - (n30 - 3*n12) * (n21 + n03) * (3*(n30 + n12)**2 - (n21 + n03)**2)

    Hu = [phi1, phi2, phi3, phi4, phi5, phi6, phi7]
    return Hu