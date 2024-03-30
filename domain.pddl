(define (domain caja-ordenar2)
    (:requirements :negative-preconditions) 
    (:predicates
        (cajaObj ?x)
        (encima ?x ?y)
        (Disponible ?x)
        (nadaSobre ?x)
        (baseCaja ?x)
        (posicionBase ?x)
        (clear ?x)
        (colocado ?x) 
    )
    (:action quitar
        :parameters (?c1 ?c2)
        :precondition (
            and
            (cajaObj ?c1)
            (cajaObj ?c2)
            (encima ?c1 ?c2)
            (nadaSobre ?c1)
            (not(baseCaja ?c1))
        )
        :effect (
            and
            (not(encima ?c1 ?c2))
            (Disponible ?c1)
            (nadaSobre ?c2)
            (not (colocado ?c1)) 
        )
    )
    (:action quitar-base
        :parameters (?c ?b)
        :precondition (
            and
            (cajaObj ?c)
            (posicionBase ?b)
            (baseCaja ?c)
            (not(clear ?b))
            (nadaSobre ?c)
        )
        :effect (
            and
            (Disponible ?c)
            (not(baseCaja ?c))
            (clear ?b)
            (not (colocado ?c)) 
        )
    )
    (:action colocar
        :parameters (?c1 ?c2)
        :precondition (
            and
            (cajaObj ?c1)
            (cajaObj ?c2)
            (Disponible ?c1)
            (nadaSobre ?c2)
            (colocado ?c2) 
        )
        :effect (
            and
            (encima ?c1 ?c2)
            (not(Disponible ?c1))
            (not(nadaSobre ?c2))
            (colocado ?c1) 
        )
    )
    (:action colocar-base
        :parameters (?c ?b)
        :precondition (
            and
            (cajaObj ?c)
            (posicionBase ?b)
            (Disponible ?c)
            (nadaSobre ?c)
            (clear ?b)
        )
        :effect (
            and
            (not(Disponible ?c))
            (baseCaja ?c)
            (not(clear ?b))
            (colocado ?c) 
        )
    )
)