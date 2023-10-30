
./image_converter.py GLADE10001-g.fits ./output/g/GLADE10001-g.npy
./image_converter.py GLADE10001-r.fits ./output/r/GLADE10001-r.npy
./image_converter.py GLADE10001-i.fits ./output/i/GLADE10001-i.npy
./image_converter.py GLADE10001-z.fits ./output/z/GLADE10001-z.npy
./image_converter.py GLADE10001-y.fits ./output/y/GLADE10001-y.npy


./mask_generator.py GLADE10001-g.fits ./output/g GLADE10001-g
./mask_generator.py GLADE10001-r.fits ./output/r GLADE10001-r
./mask_generator.py GLADE10001-i.fits ./output/i GLADE10001-i
./mask_generator.py GLADE10001-z.fits ./output/z GLADE10001-z
./mask_generator.py GLADE10001-y.fits ./output/y GLADE10001-y
