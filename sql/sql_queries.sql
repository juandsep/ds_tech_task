-- 1. Top 5 customer areas with highest average delivery time in the last 30 days.
SELECT 
    c.area AS customer_area,
    AVG(julianday(o.delivery_time) - julianday(o.order_time)) * 24 * 60 AS avg_delivery_minutes
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE o.delivery_time >= datetime('now', '-30 days')
GROUP BY c.area
ORDER BY avg_delivery_minutes DESC
LIMIT 5;

-- 2. Average delivery time per traffic condition, by restaurant area and cuisine type.
SELECT 
    o.traffic_condition,
    r.area AS restaurant_area,
    r.cuisine_type,
    AVG(julianday(o.delivery_time) - julianday(o.order_time)) * 24 * 60 AS avg_delivery_minutes
FROM orders o
JOIN restaurants r ON o.restaurant_id = r.restaurant_id
GROUP BY o.traffic_condition, r.area, r.cuisine_type;

-- 3. Top 10 delivery people with the fastest average delivery time, considering only those with at least 50 deliveries and who are still active.
SELECT 
    d.delivery_person_id,
    d.name,
    COUNT(o.order_id) AS total_deliveries,
    AVG(julianday(o.delivery_time) - julianday(o.order_time)) * 24 * 60 AS avg_delivery_minutes
FROM orders o
JOIN delivery_people d ON o.delivery_person_id = d.delivery_person_id
WHERE d.status = 'active'
GROUP BY d.delivery_person_id, d.name
HAVING COUNT(o.order_id) >= 50
ORDER BY avg_delivery_minutes ASC
LIMIT 10;

-- 4. The most profitable restaurant area in the last 3 months, defined as the area with the highest total order value.
SELECT 
    r.area AS restaurant_area,
    SUM(o.order_value) AS total_order_value
FROM orders o
JOIN restaurants r ON o.restaurant_id = r.restaurant_id
WHERE o.order_time >= datetime('now', '-3 months')
GROUP BY r.area
ORDER BY total_order_value DESC
LIMIT 1;

-- 5. Identificar repartidores con tendencia creciente en el tiempo promedio de entrega mensual.
WITH monthly_avg AS (
    SELECT
        o.delivery_person_id,
        strftime('%Y-%m', o.order_time) AS year_month,
        AVG(julianday(o.delivery_time) - julianday(o.order_time)) * 24 * 60 AS avg_delivery_minutes
    FROM orders o
    GROUP BY o.delivery_person_id, year_month
),
monthly_diff AS (
    SELECT
        delivery_person_id,
        avg_delivery_minutes,
        LAG(avg_delivery_minutes) OVER (PARTITION BY delivery_person_id ORDER BY year_month) AS prev_avg
    FROM monthly_avg
)
SELECT
    delivery_person_id
FROM monthly_diff
WHERE prev_avg IS NOT NULL AND avg_delivery_minutes > prev_avg
GROUP BY delivery_person_id;
