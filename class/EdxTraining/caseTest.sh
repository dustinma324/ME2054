!#/bin/bash

echo "Do you want to destry your entire file system?"
read response

case "$response" in

	"yes" )	echo "I hope you know what you are doing!";
		echo "I am suppose to type : rm -rf ......";;
	"no" )	echo "You have some balls to test me";;
	"y" | "Y" | "YES" ) echo "You have some balls to delete me";;
	"n" | "N" | "NO" )	echo " Dont ever delete me";;
	* )	echo "You have to give an answer!";;
esac
exit 0
