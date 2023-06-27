
import numpy as np
import operator
from functools import reduce
import matplotlib.pyplot as plt
import ase.io
import ase.calculators




def main():
    inpFile = 'in.extxyz'
    outFile = 'out.extxyz'

    # read file
    structures = ase.io.read(inpFile, index=':')

    # get a list of all elements present in the dataset
    allElems = set()
    for s in structures:
        allElems = allElems.union(set(s.get_atomic_numbers()))
    allElems = sorted(list(allElems))

    # should be a list of all elements present in the dataset
    print('found elements:', allElems)

    # build matrices with energy and number of elements in structure
    es = []  # energies
    ns = []  # number of atoms of each element
    for s in structures:
        es.append(s.get_potential_energy())
        ns.append([len([x for x in s.get_atomic_numbers() if x==el]) for el in allElems])

    # we determine the optimal atomic energies by finding the minimimal squared error solution to the following equations
    # (additionally the elements are divided by the number of atoms)
    #
    # | n_{1,el_1}, n_{1,el_2}, n_(1,el_{N_el}) |    | E_{el_1}      |          | E_1 |
    # | n_{2,el_1}, n_{2,el_2}, n_(2,el_{N_el}) |    | E_{el_1}      |          | E_2 |
    #              ...                                     ...             =      ...
    # | n_{N,el_1}, n_{N,el_2}, n_(N,el_{N_el}) |    | E_{el_{N_el}} |          | E_N |
    #

    # x, res, rnk, s = np.linalg.lstsq(np.array(ns), np.array(es), rcond=-1)
    # this would be the straight forward approach.
    # unfortunately it does not allow regularization

    # so we just implement it ourselves
    ns = np.array(ns)
    es = np.array(es)
    # divide everything by the number of atoms
    # since we want to minimize the error in the per atom energy
    # (it probably does not make a big difference)
    nats = np.sum(ns, axis=1)
    ns_per_atom = ns / nats[:, None]
    es_per_atom = es / nats[:]

    # Here we add a tiny bit of regularization
    lam = 1.e-8  # regularization parameter
    # least square solution is given by this here ...
    x = np.linalg.solve(ns_per_atom.T @ ns_per_atom + np.eye(len(allElems)) * lam, ns_per_atom.T @ es_per_atom)

    print()
    print('Optimal atomic energies:')
    for el, en in zip(allElems, x):
        print(f'    {el:02d} {en}')

    print()

    newes = es - ns @ x
    print('old mean energy per atom', np.mean(es / nats))
    print('new mean energy per atom', np.mean(newes / nats))
    print()
    print('old root mean squared energy per atom', np.sqrt(np.mean((es / nats)**2)))
    print('new root mean squared energy per atom', np.sqrt(np.mean((newes / nats)**2)))

    # histogram of the per atom energy
    # There will be multiple peaks depending on the per atom energy
    plt.hist(es / nats, bins=100)
    plt.title('energy distribution in original dataset')
    plt.xlabel('energy per atom [eV]')
    plt.tight_layout()
    plt.show()
    # now the problem should be solved
    plt.hist(newes / nats, bins=100)
    plt.title('energy distribution in modified dataset')
    plt.xlabel('energy per atom [eV]')
    plt.tight_layout()
    plt.show()

    # substract the atomic energies from the dataset
    for s in structures:
        elcount = np.array([len([x for x in s.get_atomic_numbers() if x==el]) for el in allElems])
        newenergy = s.get_potential_energy() - np.dot(elcount, x)
        s.calc.results['energy'] = newenergy

    ase.io.write(outFile, structures)


if __name__ == '__main__':
    main()
