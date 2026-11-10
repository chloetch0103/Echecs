

def calcul_lambda(lambda_: float, eta: float, d: int, danger_mes: int) -> float:
    """

    :param lambda_:
    :param eta:
    :param d:
    :param danger_mes:
    :return:
    """
    return max(0.0, lambda_+eta*(danger_mes-d))


def calcul_epsilon(epsilon_final: float, epsilon_initial: float, n: int, parties_decroissance: int) -> float:
    return max(epsilon_final, epsilon_initial - (epsilon_initial - epsilon_final) * (
                n / parties_decroissance))


def decision_curriculum(etat_curriculum: bool, retour_moyen_fenetre: float, seuil_montee_curriculum:float,
                        nbr_retour: int, fenetre: int, niveau: int) -> bool:
    return (etat_curriculum and retour_moyen_fenetre > seuil_montee_curriculum
                        and nbr_retour >= fenetre and niveau < 20)