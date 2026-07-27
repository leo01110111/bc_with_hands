ROOT    := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
PY      := $(ROOT)/.venv/bin/python

## your repo -- `student` is a symlink to it; override with
##   make check STUDENT=/path/to/wuji_bc
STUDENT ?= $(ROOT)/student

export STUDENT
export PYTHONPATH := $(ROOT)/teacher:$(STUDENT)
export MUJOCO_GL  := egl
export XLA_PYTHON_CLIENT_PREALLOCATE := false

LEVEL ?=

.PHONY: check verbose level hint peek brief demos video clean help

## run the ladder (stops at your first failing level)
check:
	@$(PY) -m katas $(if $(LEVEL),--level $(LEVEL),)

## same, but show anything your code prints
verbose:
	@$(PY) -m katas -v $(if $(LEVEL),--level $(LEVEL),)

## run exactly one level: make level LEVEL=4
level:
	@$(PY) -m katas --level $(LEVEL)

## escalating hints: make hint LEVEL=4
hint:
	@$(PY) -m katas --show hint --level $(LEVEL)

## where OGPO does this, if you want to read the reference
peek:
	@$(PY) -m katas --show peek --level $(LEVEL)

## restate the task for a level
brief:
	@$(PY) -m katas --show brief --level $(LEVEL)

## regenerate demonstrations (N=2000 for a bigger dataset)
N ?= 600
demos:
	@$(PY) -m wuji_hands.collect --episodes $(N) --noise 0.5

## watch the scripted expert
video:
	@$(PY) -c "import wuji_hands.demo_video as d; d.main()"

clean:
	@find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

help:
	@echo ""
	@echo "  make check              run the ladder"
	@echo "  make hint LEVEL=4       escalating hints for level 4"
	@echo "  make peek LEVEL=4       the corresponding OGPO source"
	@echo "  make brief LEVEL=4      restate level 4's task"
	@echo "  make level LEVEL=4      run only level 4"
	@echo "  make demos N=2000       collect more demonstrations"
	@echo "  make video              watch the scripted expert"
	@echo ""
	@echo "  teacher code  $(ROOT)/teacher"
	@echo "  your code     $(STUDENT)/wuji_bc"
	@echo ""
