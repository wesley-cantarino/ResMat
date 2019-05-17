# -*- coding: utf-8 -*-

import math

def equation(x0, *data):
    xte, xtd, yte, ytd, T0, ms = data
    return yte -ytd -T0/ms*(math.cosh(ms/T0*(xte - x0)) - math.cosh(ms/T0*(xtd - x0)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    