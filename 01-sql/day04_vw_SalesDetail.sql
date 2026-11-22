CREATE   VIEW dbo.vw_SalesDetail AS SELECT f.SalesOrderNumber, f.OrderDate, d.CalendarYear AS Yr, d.MonthNumberOfYear AS Mth, c.EnglishProductCategoryName AS Category, p.EnglishProductName AS Product, g.EnglishCountryRegionName AS Country, f.OrderQuantity,

(1 row affected)
