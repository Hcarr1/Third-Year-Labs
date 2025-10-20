package Shared;

import java.io.Serializable;

// Server.Server.Args.java (place in server/ and client/)
// This file will need a fix 
public class Args implements Serializable {
    public double a;
    public double b;

    public Args(double a, double b) {
        this.a = a;
        this.b = b;
    }
}

