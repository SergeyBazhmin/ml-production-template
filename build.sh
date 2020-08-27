#!/bin/bash

case $1 in
	test|production) docker build --target $1 -t sbazhmin/test_hello -f docker/Dockerfile .;;
	*)               echo "test/production";;
esac

