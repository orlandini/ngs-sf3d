from ngsolve import BaseMatrix


class SymmetricGS(BaseMatrix):
    def __init__(self, smoother):
        super(SymmetricGS, self).__init__()
        self.smoother = smoother

    def Mult(self, x, y):
        y[:] = 0.0
        self.smoother.Smooth(y, x)
        self.smoother.SmoothBack(y, x)

    def Height(self):
        return self.smoother.height

    def Width(self):
        return self.smoother.height


def ElPatchBlocks(mesh, fes):
    blocks = []
    freedofs = fes.FreeDofs()
    for el in mesh.Elements():
        eldofs = set()
        eldofs |= set(d for d in fes.GetDofNrs(el) if freedofs[d])
        blocks.append(eldofs)
    # for e in mesh.edges:
    #     edofs = set()
    #     for el in mesh[e].elements:
    #         edofs |= set(d for d in fes.GetDofNrs(el)
    #                      if freedofs[d])
    #     blocks.append(edofs)
    # for v in mesh.vertices:
    #     vdofs = set()
    #     for el in mesh[v].elements:
    #         vdofs |= set(d for d in fes.GetDofNrs(el)
    #                      if freedofs[d])
    #     blocks.append(vdofs)
    return blocks


def CreatePrecond(mesh, fes, a):
    print("creating blocks")
    blocks = ElPatchBlocks(mesh, fes)
    print("created {} blocks!\nsmoothing...".format(len(blocks)))
    blockjac = a.mat.CreateBlockSmoother(blocks)
    print("\rsmoothed!Creating pre...")
    pre = SymmetricGS(blockjac)
    print("\rCreated pre!")
    return pre
