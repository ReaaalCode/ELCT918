`timescale 1ns/1ps
module tb;
  reg clk=0, rst_n=0, en=0;
  reg [3:0] din=0;
  wire [7:0] ref_dout, sol_dout;
  wire ref_flag, sol_flag;
  integer cycles=0, mismatches=0, i;
  integer a, b, c, d;
  reg [31:0] rv;
  reg found;
  reg [3:0] password [0:3];

  mystery ref_chip(.clk(clk),.rst_n(rst_n),.en(en),.din(din),.dout(ref_dout),.flag(ref_flag));
  solution dut(.clk(clk),.rst_n(rst_n),.en(en),.din(din),.dout(sol_dout),.flag(sol_flag));
  always #5 clk=~clk;

  task cycle(input reg reset_value, input reg enable_value, input reg [3:0] data_value);
    begin
      @(negedge clk);
      rst_n=reset_value; en=enable_value; din=data_value;
      @(posedge clk); #1;
      cycles=cycles+1;
      if ((ref_dout !== sol_dout) || (ref_flag !== sol_flag)) begin
        mismatches=mismatches+1;
        if (mismatches <= 10)
          $display("MISMATCH cycle=%0d rst_n=%b en=%b din=%h reference=%h/%b solution=%h/%b",
                   cycles,rst_n,en,din,ref_dout,ref_flag,sol_dout,sol_flag);
      end
    end
  endtask

  initial begin
    $dumpfile("dump.vcd"); $dumpvars(0,tb);
    cycle(0,0,0); cycle(0,0,0);
    $display("After synchronous reset: dout=%h flag=%b",ref_dout,ref_flag);

    // Directed experiments: disabled inputs hold the low output nibble;
    // repeated enabled inputs reveal state dependence.
    cycle(1,0,5); $display("en=0 din=5 -> dout=%h flag=%b",ref_dout,ref_flag);
    cycle(1,0,9); $display("en=0 din=9 -> dout=%h flag=%b",ref_dout,ref_flag);
    cycle(1,1,5); $display("en=1 din=5 -> dout=%h flag=%b",ref_dout,ref_flag);
    cycle(1,1,5); $display("repeat en=1 din=5 -> dout=%h flag=%b",ref_dout,ref_flag);

    // At least 10,000 random cycles; reset, enable, and data are randomized.
    for (i=0;i<10000;i=i+1) begin
      rv=$random;
      cycle((rv[6:0] != 0),rv[8],rv[12:9]);
    end

    // Search every four-nibble plaintext sequence from a fresh reset.
    found=0;
    for (a=0; a<16 && !found; a=a+1)
      for (b=0; b<16 && !found; b=b+1)
        for (c=0; c<16 && !found; c=c+1)
          for (d=0; d<16 && !found; d=d+1) begin
            cycle(0,0,0); cycle(0,0,0);
            cycle(1,1,a[3:0]);
            cycle(1,1,b[3:0]);
            cycle(1,1,c[3:0]);
            cycle(1,1,d[3:0]);
            if (ref_flag) begin
              found=1;
              password[0]=a[3:0]; password[1]=b[3:0];
              password[2]=c[3:0]; password[3]=d[3:0];
              $display("Found password: %h%h%h%h",password[0],password[1],password[2],password[3]);
            end
          end

    if (found) begin
      // Reset, then replay the discovered sequence to demonstrate it.
      cycle(0,0,0); cycle(0,0,0);
      $display("Replaying the discovered password after reset");
      for (i=0; i<4; i=i+1) begin
        cycle(1,1,password[i]);
        $display("din=%h dout=%h flag=%b",password[i],ref_dout,ref_flag);
      end
    end else begin
      $display("FAIL: no four-nibble sequence asserted flag");
    end

    if ((mismatches==0) && found) $display("PASS: %0d cycles compared, zero mismatches",cycles);
    else $display("FAIL: %0d mismatches in %0d cycles; password_found=%b",mismatches,cycles,found);
    $finish;
  end
endmodule
