# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:50:21 2019

@author: Guilherme Márcio
"""
import math

def equation(x0, *data):
    xte,xtd,yte,ytd,T0,ms = data
    return yte -ytd -T0/ms*(math.cosh(ms/T0*(xte-x0))-math.cosh(ms/T0*(xtd-x0)))
    