set total_frames [molinfo top get numframes]

for {set frame 0} {$frame < $total_frames} {incr frame} {
    animate goto $frame 

    set com {0 0 0}
    set count 0

    for { set x 0 } { $x < 46 } { incr x } {
        set sel [atomselect $x all]
        foreach coord [$sel get {x y z}] {
            set com [vecadd $com $coord]
            incr count
        }
    }

    set com [vecscale $com [expr 1 / $count]]

    for { set x 0 } { $x < 46 } { incr x } {
        set sel [atomselect $x all]
        $sel moveby [vecsub {0 0 0} $com]
    }
}

for {set x 0} {$x <= 45} {incr x} {

    mol modstyle 0 $x Tube 0.50000 20.000000
    mol modselect 0 $x resname ASP and z < 0
    mol modcolor 0 $x ColorID 1
    mol modmaterial 0 $x AOChalky

    mol modstyle 1 $x Tube 0.50000 20.000000
    mol modselect 1 $x resname GLU and z < 0
    mol modcolor 1 $x ColorID 3
    mol modmaterial 1 $x AOChalky

    mol modstyle 2 $x Tube 0.50000 20.000000
    mol modselect 2 $x resname HIS and z < 0
    mol modcolor 2 $x ColorID 0
    mol modmaterial 2 $x AOChalky

    mol modstyle 3 $x Tube 0.50000 20.000000
    mol modselect 3 $x resname LYS and z < 0
    mol modcolor 3 $x ColorID 10
    mol modmaterial 3 $x AOChalky

    mol modstyle 4 $x Tube 0.50000 20.000000
    mol modselect 4 $x resname ARG and z < 0
    mol modcolor 4 $x ColorID 7
    mol modmaterial 4 $x AOChalky

    mol modstyle 5 $x Tube 0.50000 20.000000
    mol modselect 5 $x resname ASN and z < 0
    mol modcolor 5 $x ColorID 2
    mol modmaterial 5 $x AOChalky

}
color change rgb 7 0.280000 0.500000 0.120000

for {set x 0} {$x <= 45} {incr x} {

    mol modstyle 0 $x Tube 0.50000 20.000000
    mol modselect 0 $x resname ASP GLU and z < 0
    mol modcolor 0 $x ColorID 1
    mol modmaterial 0 $x AOChalky

    mol modstyle 1 $x Tube 0.50000 20.000000
    mol modselect 1 $x resname HIS LYS ARG and z < 0
    mol modcolor 1 $x ColorID 0
    mol modmaterial 1 $x AOChalky

    mol modstyle 2 $x Tube 0.50000 20.000000
    mol modselect 2 $x resname ASN and z < 0
    mol modcolor 2 $x ColorID 2
    mol modmaterial 2 $x AOChalky

}