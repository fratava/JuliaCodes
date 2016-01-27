## module run_kut5
## 
using PyCall
@pyimport numpy

function integrate(F,x,y,xStop,h,tol=1.0e-6)
    
    function run_kut5(F,x,y,h)
    # Runge--Kutta-Fehlberg formulas
        C = ([37./378, 0., 250./621, 125./594,0., 512./1771])
        D = ([2825./27648, 0., 18575./48384,13525./55296, 277./14336, 1./4])
        n = length(y)
        #M=K
        M = zeros((6,n))
        M[1,1],M[1,2] = h*F(x,y)
        y1=([M[1,1],M[1,2]])
        M[2,1],M[2,2] = h*F(x + 1./5*h, y+1/5*y1)
        y2=([M[2,1],M[2,2]])
        M[3,1],M[3,2] = h*F(x + 3/10*h, y + 3/40*y1 + 9/40*y2)
        y3=([M[3,1],M[3,2]])
        M[4,1],M[4,2] = h*F(x + 3/5*h, y + 3/10*y1- 9/10*y2+ 6/5*y3)
        y4=([M[4,1],M[4,2]])
        M[5,1],M[5,2] = h*F(x + h, y - 11./54*y1 + 5./2*y2- 70./27*y3 + 35./27*y4)
        y5=([M[5,1],M[5,2]])
        M[6,1],M[6,2] = h*F(x + 7./8*h, y + 1631./55296*y1+ 175./512*y2 
        + 575./13824*y3+ 44275./110592*y4 
        + 253./4096*y5)
        y6=([M[6,1],M[6,2]])
        # Initialize arrays {dy} and {E}
        E = zeros(n)
        dy = zeros(n)
        # Compute solution increment {dy} and per-step error {E}
       for i in 1:6
            dy1=([C[i]*M[i,1],C[i]*M[i,2]])
            dy = dy + dy1
            E1=([(C[i] - D[i])*M[i,1],(C[i] - D[i])*M[i,2]])
            E = E + E1        
        end
        # Compute RMS error e
        e =0
        for i in 1:length(E)
            e=sqrt((E[i]^2)/n)
        end
        return dy,e
    end
    
    
    X = []
    Y = []
    push!(X,x)
    push!(Y,y)
    stopper = 0 # Integration stopper(0 = off, 1 = on)
    
    for i in 1:10000
        dy,e = run_kut5(F,x,y,h)
        # Accept integration step if error e is within tolerance
        if e <= tol
            y = y + dy
            x = x + h
            push!(X,x)
            push!(Y,y)
            # Stop if end of integration range is reached
            if stopper == 1
                break
            end
        end
        # Compute next step size from Eq. (7.24)
        if e != 0.0
            hNext = 0.9*h*(tol/e)^0.2
        else
            hNext = h
        end
        # Check if next step is the last one; is so, adjust h
        if (h > 0.0) == ((x + hNext) >= xStop)
            hNext = xStop - x
            stopper = 1
        end
        h = hNext
    end
    return numpy.array(X),numpy.array(Y)
end
