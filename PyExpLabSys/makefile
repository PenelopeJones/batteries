test2:
	python2 -m pytest

test3:
	python3 -m pytest

docs:
	$(MAKE) -C doc html
	@echo "\033[95m\n\nBuild successful! View the docs at docs/_build/html/index.html.\n\033[0m"

clean:
	find . -name '*.py[co]' -delete

	find . -name '*~' -delete
	find . -name '__pycache__' -delete
	rm -rf soco.egg-info
	rm -rf dist
	$(MAKE) -C doc clean

.PHONY: docs test clean
