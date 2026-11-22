CREATE   PROCEDURE dbo.usp_TopCustomers @Country NVARCHAR(50), @TopN INT = 10 AS BEGIN SET NOCOUNT ON; SELECT TOP (@TopN) cu.CustomerKey, cu.FirstName + ' ' + cu.LastName AS Customer, SUM(f.SalesAmount) AS Revenue FROM dbo.FactInternetSales f JOIN dbo.DimCustomer cu ON f.CustomerKey = cu.CustomerKey JOIN dbo.DimGeography g ON cu.GeographyKey = g.GeographyKey WHERE g.EnglishCountryRegionName = @Country GROUP BY cu.CustomerKey, cu.FirstName, cu.LastName ORDER BY Revenue DESC; END

(1 row affected)
