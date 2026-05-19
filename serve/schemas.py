"""Pydantic schemas for the prediction API."""
from typing import Optional

from pydantic import BaseModel, Field


class ClienteInput(BaseModel):
    LINEA_RENOVADO: float = Field(..., gt=0, description="Monto de línea ofrecido (S/)")
    PLAZO_RENOVADO: int = Field(..., ge=6, le=60, description="Plazo en meses")
    USO_LINEA_TOTAL_TC_T2: Optional[float] = Field(None, description="Uso % línea TC total")
    USO_TRIM_LINEA_BBVA: Optional[float] = Field(None, description="Uso % línea TC banco")
    NR_ENTIDADES_TOTAL_T2: int = Field(..., ge=0, description="N° entidades financieras")
    DIFF_NRO_ENTIDA_TOTALES_T2_T12: int = Field(0, description="Variación anual en entidades")
    SDO_CONSUMO_T2: Optional[float] = Field(None, description="Saldo consumo")
    RESENCIA_OFERTA_PLD_RENOVADO: Optional[float] = Field(None, description="Meses desde primera oferta")
    Ahorro_Sldo_Bco_T1: float = Field(0.0, description="Saldo ahorro en banco")
    PConsumo_Sldo_Bco_T1: float = Field(0.0, description="Saldo préstamo activo en banco")
    SDO_BCO_tot_sm_pasivo_Bco_6M: float = Field(..., description="Promedio deuda 6 meses")
    EDAD: float = Field(..., ge=18, le=100, description="Edad del cliente")
    SEXO: int = Field(..., ge=0, le=1, description="Sexo: 1=Masculino, 0=Femenino")
    ANTIGUEDAD_MES: float = Field(..., ge=0, description="Antigüedad en meses")
    FLAG_LIMA_PROVINCIA: int = Field(..., ge=0, le=1, description="1=Lima, 0=Provincia")
    SUELDO_ESTIMADO: Optional[float] = Field(None, description="Sueldo estimado (S/)")
    CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD: float = Field(..., description="% cobertura deuda con renovación")
    # One-hot encoded fields (generated during preprocessing)
    EST_CIVIL_DIVORCIADO: int = Field(0, ge=0, le=1)
    EST_CIVIL_SEPARADO: int = Field(0, ge=0, le=1)
    EST_CIVIL_SOLTERO: int = Field(0, ge=0, le=1)
    EST_CIVIL_UNION_LIBRE: int = Field(0, ge=0, le=1)
    EST_CIVIL_VIUDO: int = Field(0, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "LINEA_RENOVADO": 15000,
                "PLAZO_RENOVADO": 36,
                "USO_LINEA_TOTAL_TC_T2": 0.45,
                "USO_TRIM_LINEA_BBVA": 0.30,
                "NR_ENTIDADES_TOTAL_T2": 3,
                "DIFF_NRO_ENTIDA_TOTALES_T2_T12": 0,
                "SDO_CONSUMO_T2": 5000.0,
                "RESENCIA_OFERTA_PLD_RENOVADO": 6.0,
                "Ahorro_Sldo_Bco_T1": 1200.0,
                "PConsumo_Sldo_Bco_T1": 8000.0,
                "SDO_BCO_tot_sm_pasivo_Bco_6M": 950.0,
                "EDAD": 42.0,
                "SEXO": 1,
                "ANTIGUEDAD_MES": 120.0,
                "FLAG_LIMA_PROVINCIA": 1,
                "SUELDO_ESTIMADO": 4500.0,
                "CUBRIR_DEUDA_CONSUMO_SF_RENOVA_PLD": 0.65,
                "EST_CIVIL_DIVORCIADO": 0,
                "EST_CIVIL_SEPARADO": 0,
                "EST_CIVIL_SOLTERO": 0,
                "EST_CIVIL_UNION_LIBRE": 0,
                "EST_CIVIL_VIUDO": 0,
            }
        }


class PrediccionOutput(BaseModel):
    renueva: int = Field(..., description="1=Renueva, 0=No renueva")
    probabilidad_renovacion: float = Field(..., description="Probabilidad de renovación [0-1]")
    modelo: str = Field(..., description="Nombre del modelo usado")


class HealthResponse(BaseModel):
    status: str
    modelo: str
    recall_cv: Optional[float] = None
