# -*- coding: utf-8 -*-
"""
Received Power Calculator

@author: Jose Matamoros
"""
from math import log

def receivedPower(pt,d,hb):
    '''
    Calculates the received power based on the Okamura-Hata model
    
    Parameters
    ----------
    pt : float
        Transmitted Power[dB]
    d : float
        Distance[m]
    hb: float
        Base Station height[m]
        
    Returns
    -------
    float
        Received Power [dB]
    
    '''
    
    f = 700 #Carrier Frequency = 700 MHz
    hm = 2  #Mobile height set to 2 meters
    
    am = (1.1*log(f,10)-0.7)*hm - (1.56*log(f,10)-0.8)
    a = 69.55 + 26.16*log(f,10) - 13.82*log(hb,10) - am
    b = 44.9 - 6.55*log(hb,10)
    c = 0
    pl = a + b*log(d,10) + c #Path loss calculation

    
    #pl = (44.9-6.55*log(hb,10))*log(d,10)+5.83*log(hb,10)+16.33 + 26.16*log(f,10)
    
    return (pt - pl) #Return Received Power = Transmitted Power - Path Loss

if __name__ == "__main__":
	##Code below used solely for testing
    print(receivedPower(0, 0.04, 2))