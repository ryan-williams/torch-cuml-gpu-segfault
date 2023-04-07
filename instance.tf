variable "awsprops" {
  type = map
  default = {
    region = "us-east-1"
    ami = "ami-058e8127e717f752b"  # PyTorch-1.13, TensorFlow-2.11, MXNet-1.9, Neuron, & others. NVIDIA CUDA, cuDNN, NCCL, Intel MKL-DNN, Docker, NVIDIA-Docker & EFA support. For fully managed experience, check: https://aws.amazon.com/sagemaker
    itype = "p3.2xlarge"
    publicip = true
    keyname = "gpu-segfault-test-node"
    gpu-segfault-sg = "gpu-segfault-sg"
  }
}

provider "aws" {
  region = lookup(var.awsprops, "region")
}

resource "aws_security_group" "gpu-segfault-sg" {
  name = lookup(var.awsprops, "gpu-segfault-sg")
  description = lookup(var.awsprops, "gpu-segfault-sg")

  // To Allow SSH Transport
  ingress {
    from_port = 22
    protocol = "tcp"
    to_port = 22
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port       = 0
    to_port         = 0
    protocol        = "-1"
    cidr_blocks     = ["0.0.0.0/0"]
  }

  lifecycle {
    create_before_destroy = true
  }
}


resource "aws_instance" "gpu-segfault-test-instance" {
  ami = lookup(var.awsprops, "ami")
  instance_type = lookup(var.awsprops, "itype")
  associate_public_ip_address = lookup(var.awsprops, "publicip")
  key_name = lookup(var.awsprops, "keyname")

  vpc_security_group_ids = [
    aws_security_group.gpu-segfault-sg.id
  ]
  root_block_device {
    delete_on_termination = true
    iops = 150
    volume_size = 180
    volume_type = "gp3"
  }
  tags = {
    Name ="gpu-segfault-test-instance"
  }

  depends_on = [ aws_security_group.gpu-segfault-sg ]
}


output "ec2instance" {
  value = aws_instance.gpu-segfault-test-instance.public_ip
}
