import dit
from dit import Distribution
from dit.multivariate import PID_GK

# Definindo a distribuição conjunta das variáveis (X0, X1, Y)
# Y = X0 XOR X1

data = [
    ((0, 0, 0), 0.25),
    ((0, 1, 1), 0.25),
    ((1, 0, 1), 0.25),
    ((1, 1, 0), 0.25),
]

# Criando a distribuição no dit
d = Distribution(data)
d.set_rv_names(['X0', 'X1', 'Y'])

# Partial Information Decomposition (Griffith-Koch method)
pid = PID_GK(d, ['X0', 'X1'], 'Y')

# 🔍 Mostrando os componentes da decomposição
print("Partial Information Decomposition (PID):")
print(f"  Redundant info     : {pid.get_partial('redundant'):.3f}")
print(f"  Unique info X0     : {pid.get_partial(('X0',)):.3f}")
print(f"  Unique info X1     : {pid.get_partial(('X1',)):.3f}")
print(f"  Synergistic info   : {pid.get_partial('synergy'):.3f}")
print()

# 🔎 Comparando com a informação mútua total
from dit.multivariate import mutual_information as MI

mi = MI(d, ['X0', 'X1'], 'Y')
print(f"Total Mutual Information I([X0,X1]; Y): {mi:.3f}")
