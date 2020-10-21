# coding: utf-8

# In[19]:

import numpy as np
import sys
import pandas as pd
import numpy
import itertools
import toyplot
import toyplot.pdf
#get_ipython().run_line_magic('matplotlib', 'inline')


filename = sys.argv[1]
col_0_name = "numPairs"
y_label_0 = "Insertion Throughput (billions pairs/sec)"
x_label_0 = "Load Factor"
title_0 = "K40, "
y_label_1 = "Query Throughput (billions pairs/sec)"
x_label_1 = "Load Factor"
title_1 = "K40 Querying"


df = pd.read_csv(filename, header=None, names=['numPairs0', 'hfInsert00', 'hfQuery00', 'cuInsert00', 'cuQuery00', 'thSort00', 'thSearch00', 'hfInsert01', 'hfQuery01', 'cuInsert01', 'cuQuery01', 'thSort01', 'thSearch01', 'hfInsert10', 'hfQuery10', 'cuInsert10', 'cuQuery10', 'thSort10', 'thSearch10', 'hfInsert11', 'hfQuery11', 'cuInsert11', 'cuQuery11', 'thSort11', 'thSearch11'])

df['numPairs0'] = df.numPairs0*100

df['hfQuery00'] = 1450000000/df.hfQuery00
df['cuQuery00'] = 1450000000/df.cuQuery00
df['thSearch00'] = 1450000000/df.thSearch00
df['hfQuery01'] = 1150000000/df.hfQuery01
df['cuQuery01'] = 1150000000/df.cuQuery01
df['thSearch01'] = 1150000000/df.thSearch01

df['hfQuery10'] = 500000000/df.hfQuery10
df['cuQuery10'] = 500000000/df.cuQuery10
df['thSearch10'] = 500000000/df.thSearch10
df['hfQuery11'] = 400000000/df.hfQuery11
df['cuQuery11'] = 400000000/df.cuQuery11
df['thSearch11'] = 400000000/df.thSearch11

colors = ["green", "steelblue", "darkred",
          "green", "steelblue", "darkred"]
markers = ["o", "--", "^",
           "o", "--", "^"]
exts = ["", "2"]

def GetStyle(toplot) :
  style = {}
  keys = algos.keys()
  for key in keys:
    if key in toplot :
      index = algos.get(key)
      style["stroke"] = colors[index]
    if "01" in toplot or "11" in toplot :
      style["stroke-dasharray"] = "2, 2"
  return style


canvas = toyplot.Canvas('3.5in', '6in')

#Plot 2: K40 Queries
algos = {"hfQuery10":0, "cuQuery10":1, "thSearch10":2,
         "hfQuery11":3, "cuQuery11":4, "thSearch11":5}
legends = {"hfQuery10":"HashFight,1.03,500M", \
           "cuQuery10":"CUDPP,1.03,500M", \
           "thSearch10":"Thrust-Search,500M", \
           "hfQuery11":"HashFight,1.50,400M", \
           "cuQuery11":"CUDPP,1.50,400M", \
           "thSearch11":"Thrust-Search,400M"}

axes2 = canvas.cartesian(label = title_1,
                        xlabel = 'Failed Queries (%)',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid=(2,1,0),
                        margin = (40,50,30))

#axes2.label.location = "above"
#axes2.label.offset = "-20px"
axes2.label.style = {"font-size":"12px"}
axes2.x.label.style = {"font-size":"8px"}
axes2.y.label.style = {"font-size":"8px"}
axes2.x.ticks.show = True
axes2.y.ticks.show = True
axes2.y.ticks.labels.angle = -90
axes2.x.ticks.far = "4px"
axes2.y.ticks.far = "4px"
axes2.y.label.location = "above"
axes2.y.label.offset = "30px"

axes2.x.ticks.locator = toyplot.locator.Explicit([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
axes2.y.ticks.locator = toyplot.locator.Explicit([0, 50, 100, 150, 200, 250, 300, 350, 400, 450])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    x = df.numPairs0.values
    y = numpy.array(data)
    plotstyle = GetStyle(toplot)
    label = legends.get(toplot)
    thisMarker = markers[algos.get(algo)]
    thisStroke = plotstyle.get("stroke")
    axes2.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes2._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "bottom"
    if "thSearch10" in algo:
      vertalign = "top"
    if "thSearch11" in algo:
      vertalign = "middle"

    axes2.text(x[-1], y[-1], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})


#Plot 4: V100 Queries
algos = {"hfQuery00":0, "cuQuery00":1, "thSearch00":2,
         "hfQuery01":3, "cuQuery01":4, "thSearch01":5}
legends = {"hfQuery00":"HashFight,1.03,1450M", \
           "cuQuery00":"CUDPP,1.03,1450M", \
           "thSearch00":"Thrust-Search,1450M", \
           "hfQuery01":"HashFight,1.50,1150M", \
           "cuQuery01":"CUDPP,1.50,1150M", \
           "thSearch01":"Thrust-Search,1150M"}

axes4 = canvas.cartesian(label = "V100 Querying",
                        xlabel = 'Query Failure Rate (%)',
                        ylabel = 'Query Throughput (1M pairs/sec)',
                        grid = (2,1,1),
                        margin = (30,50,50))

axes4.label.location = "above"
axes4.label.offset = "-20px"
axes4.label.style = {"font-size":"12px"}
axes4.x.label.style = {"font-size":"8px"}
axes4.y.label.style = {"font-size":"8px"}
axes4.x.ticks.show = True
axes4.y.ticks.show = True
axes4.y.ticks.labels.angle = -45
axes4.x.ticks.far = "4px"
axes4.y.ticks.far = "4px"
axes4.y.label.location = "above"
axes4.y.label.offset = "30px"

#axes4.y.domain.max = 4000
axes4.x.ticks.locator = toyplot.locator.Explicit([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
axes4.y.ticks.locator = toyplot.locator.Explicit([0, 500, 1000, 1500, 2000, 2500])

for algo in algos.keys() :
  #for ext in exts :
    #toplot = "%s%s" %(algo, ext)
    toplot = algo
    data = df[toplot]
    data = data / 1000000.0
    x = df.numPairs0.values
    y = numpy.array(data)
    plotstyle = GetStyle(toplot)
    label = legends.get(toplot)
    thisMarker = markers[algos.get(algo)]
    thisStroke = plotstyle.get("stroke")
    axes4.plot(x, y, style=plotstyle, marker=thisMarker, mstyle={"stroke":thisStroke, "fill":thisStroke})
    print "stroke : %s" %plotstyle.get("stroke")
    axes4._text_colors = itertools.cycle([plotstyle.get("stroke")])
    vertalign = "top"
    if "cuQuery00" in algo:
      vertalign = "bottom"
    if "thSearch00" in algo:
      vertalign = "bottom"

    axes4.text(x[-1], y[-1], label, style={"text-anchor":"start", \
                                          "-toyplot-anchor-shift":"2.5px", \
                                          "font-size":"6px", \
                                          "-toyplot-vertical-align":vertalign})

canvas.aspect = 'fit-range'
toyplot.pdf.render(canvas, "vary-query.pdf")
