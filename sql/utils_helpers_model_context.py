import polars as pl
import pandas as pd


def rename_cols(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rename columns in a Polars DataFrame to standardized names.
    """
    return df.rename(
        {
            "Salesman_Brand": "brand",
            "Salesman_Branch": "branch",
            "Designation": "designation",
            "SKU_Code": "sku_code",
            "Salesman_Code": "salesman_code",
            "Salesman_Name": "salesman_name",
            "Customer_Name": "customer_name",
            "Customer_Phone": "customer_phone",
            "Order_Id": "order_id",
            "Delivered_date": "date",
            "Delivered_Qty": "quantity",
            "Delivered_Value": "value",
            "SKU_Name": "sku_name",
            "Order_Qty": "order_qty",
            "Order_Value": "order_value",
            "UOM": "uom",
            "Customer_Code": "customer_code",
            "Customer_City": "customer_city",
            "Customer_State": "customer_state",
            "Customer_Latitude": "customer_latitude",
            "Customer_Longitude": "customer_longitude",
        }
    )


def rename_cols_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns in a pandas DataFrame to standardized names.
    """
    column_map = {
        "Salesman_Brand": "brand",
        "Salesman_Branch": "branch",
        "Designation": "designation",
        "SKU_Code": "sku_code",
        "Salesman_Code": "salesman_code",
        "Salesman_Name": "salesman_name",
        "Customer_Name": "customer_name",
        "Customer_Phone": "customer_phone",
        "Order_Id": "order_id",
        "Delivered_date": "date",
        "Delivered_Qty": "quantity",
        "Delivered_Value": "value",
        "SKU_Name": "sku_name",
        "Order_Qty": "order_qty",
        "Order_Value": "order_value",
        "UOM": "uom",
        "Customer_Code": "customer_code",
        "Customer_City": "customer_city",
        "Customer_State": "customer_state",
        "Customer_Latitude": "customer_latitude",
        "Customer_Longitude": "customer_longitude",
    }
    return df.rename(columns=column_map)
