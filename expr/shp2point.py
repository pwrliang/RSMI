#!/usr/bin/env python3
import geopandas
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

if len(sys.argv) != 3:
    exit(1)
df = geopandas.read_file(in_file)
df = df.to_crs(epsg=4326)

min_x, min_y, max_x, max_y = df.total_bounds

with open(out_file, 'w') as fo:
    pid = 0

    for id, geometry in df[['geometry']].iterrows():
        geometry = geometry[0]
        x = None
        y = None

        if geometry.geom_type == 'Polygon':
            for p in geometry.exterior.coords:
                x, y = p
                pid += 1
        else:  # MultiPolygon
            for poly in geometry.geoms:
                for p in poly.exterior.coords:
                    x, y = p
                    pid += 1

        if x is not None:
            x = (x - min_x) / (max_x - min_x)
            y = (y - min_y) / (max_y - min_y)
            fo.write("%.9f,%.9f,%d\n" % (y, x, pid))
