import pandas as pd

def Poisson_ratio () :
    
    file_path = 'Rock Elastic Moduli for Well B.xlsx'
    df = pd.read_excel(file_path)
    Vs = df.iloc[1:2487, 5]
    Vp = df.iloc[1:2487, 6]
    nu = 0.5 * ((Vp**2 - 2*Vs**2) / (Vp**2 - Vs**2))
    df.iloc[1:2487, 8] = nu
    df.to_excel(file_path, index=False)
    print("Calculation completed and results saved!")

Poisson_ratio ()

def Shear_Modulus () :
    file_path = 'Rock Elastic Moduli for Well B.xlsx'
    df = pd.read_excel(file_path)
    DTs = df.iloc[1:2487, 7].values
    RHOB = df.iloc[1:2487, 1].values
    G = (RHOB / (DTs ** 2)) * 1.34 * 10**10
    df.iloc[1:2487, 9] = G
    df.to_excel(file_path, index=False)
    print("Calculation completed and results saved!")
    
Shear_Modulus ()

def Young_Modulus () :

    file_path = 'Rock Elastic Moduli for Well B.xlsx'
    df = pd.read_excel(file_path)
    DTs = df.iloc[1:2487, 7].values
    RHOB = df.iloc[1:2487, 1].values
    DT = df.iloc[1:2487, 2].values
    E_psi = (RHOB / DTs**2) * ((3 * DTs**2 - 4 * DT**2) / (DTs**2 - DT**2)) * 1.34e10
    df.iloc[1:2487, 10] = E_psi
    df.to_excel(file_path, index=False)
    print("Calculation completed and results saved!")
    
Young_Modulus ()    

def Bulk_Modulus () : 

    file_path = 'Rock Elastic Moduli for Well B.xlsx'
    df = pd.read_excel(file_path)
    DTs = df.iloc[1:2487, 7].values
    RHOB = df.iloc[1:2487, 1].values
    DT = df.iloc[1:2487, 2].values
    K_psi = RHOB * ((1 / DT**2) - (4 / (3 * DTs**2))) * 1.34 * 10**10
    df.iloc[1:2487, 11] = K_psi
    df.to_excel(file_path, index=False)
    print("Calculation completed and results saved!")

Bulk_Modulus ()

def UCS () :
    
    file_path = 'Rock Elastic Moduli for Well B.xlsx'
    df = pd.read_excel(file_path)
    def calculate_ucs(dt):
        return 0.77 * (304.8 / dt) ** 2.93
    UCS = df.iloc[1:2487, 2].apply(calculate_ucs)
    df.iloc[1:2487, 12] = UCS
    df.to_excel(file_path, index=False)
    print("Calculation completed and results saved!")
    
UCS ()    
