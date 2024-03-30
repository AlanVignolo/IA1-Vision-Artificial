(define (problem caja-ordenar-problema)
    (:domain caja-ordenar2)
    (:objects
        arandelas
        tuercas
        clavos
        tornillos
        base1
    )
    (:init
        (cajaObj arandelas)
        (cajaObj tuercas)
        (cajaObj clavos)
        (cajaObj tornillos)
        (posicionBase base1)
        (encima arandelas tornillos)
        (encima tornillos tuercas)
        (encima tuercas clavos)
        (baseCaja clavos)
        (nadaSobre arandelas)
        (not(clear base1))
    )
    (:goal (
        and
        (encima tornillos arandelas)
        (encima arandelas tuercas)
        (encima tuercas clavos)
        (baseCaja clavos)
        (colocado tornillos)
        (colocado arandelas)
        (colocado tuercas)
        (colocado clavos)
    ))
)